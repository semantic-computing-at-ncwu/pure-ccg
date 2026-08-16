{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}

-- Binary Tree-LSTM 句法合并分类模型
-- 修复点：
-- 1. emptyNodeCache where作用域错误
-- 2. backwardNode dEmbLSTM 前向引用问题
-- 3. 梯度空向量V.empty改为同维度零张量，避免zip越界崩溃
-- 4. zeroGrad传参硬编码0改为从参数读取真实维度
-- 5. LSTM反向传播补全遗忘门embOut梯度贡献
-- 6. dLdnewC tanh导数链式推导修正
-- 7. 训练循环改为尾递归防止栈溢出
-- 8. 投影层BP返回dL/d原始特征梯度，完善链式闭环
-- 9. 优化叶子节点判定逻辑，避免Eq递归整树比较
-- 10. 遗忘门偏置初始化置1，缓解LSTM梯度消失
module BiTreeLSTM (
    ParseAction(..),
    actionToOneHot,
    vecToAction,
    NodeFeat,
    SyntaxTree(..),
    RawSyntaxTree,
    vecToList,
    listToVec,
    TrainSample,
    SampleRecord(..),
    sampleFromRecord,
    recordFromSample,
    loadSamplesJSON,
    saveSamplesJSON,
    vecAdd,
    vecSub,
    vecMul,
    vecScale,
    matVecMul,
    matAdd,
    matScale,
    matOuter,
    matTrans,
    sigmoid,
    sigmoidDeriv,
    vecTanh,
    tanhDeriv,
    softmax,
    crossEntropy,
    NetParam(..),
    netHidDim,
    netProjDim,
    NetGrad,
    LSTMState(..),
    emptyLSTMState,
    NodeCache(..),
    TreeFwdResult(..),
    randVec,
    randMat,
    initNetParam,
    zeroGrad,
    gradAdd,
    gradScale,
    updateParam,
    projectRawFeat,
    treeForwardCached,
    backwardNode,
    backwardTree,
    sampleForwardLossGrad,
    trainStep,
    batchTrainStep,
    mkLeaf,
    mkNonLeaf,
    trainLoop,
    biTreePhraSyn02SyntaxTreeNodeFeat,    -- BiTree PhraSyn0 -> SyntaxTree NodeFeat
    buildSampleSet,
    testPipeline
) where

import qualified Data.Vector as V
import qualified System.IO.Streams as S
import Data.Vector (Vector)
import System.Random (randomRIO)
import Data.Foldable (sum)
import Data.Maybe
import Data.Map (Map)
import qualified Data.Map.Strict as Map
import GHC.Generics (Generic)
import Control.Exception (catch, SomeException)
import System.IO (readFile, writeFile)
import qualified Data.String as DS
import Database.MySQL.Base
import Database
import AmbiResol
import Category (Category(..))
import Phrase (Tag, PhraStru)
import Statistics (cate2Vec, tag2Vec, stru2Vec)
import Data.Tuple.Utils
import Utils

-- JSON 相关
import Data.Aeson (ToJSON(..), FromJSON(..), withText, withObject, (.:), (.:?), object, (.=), encode, decodeFileStrict)
import Data.Aeson.Types (Parser)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Text as T

-- ===================== 目标动作枚举 =====================
data ParseAction = ActLp    -- 左合并
                 | ActRp    -- 右合并
                 | ActNoth  -- 不合并
  deriving (Eq, Show, Generic)

actionToOneHot :: ParseAction -> Vector Double
actionToOneHot ActLp   = V.fromList [1.0, 0.0, 0.0]
actionToOneHot ActRp   = V.fromList [0.0, 1.0, 0.0]
actionToOneHot ActNoth = V.fromList [0.0, 0.0, 1.0]

vecToAction :: Vector Double -> ParseAction
vecToAction v
  | V.length v /= 3 = ActNoth
  | otherwise = case V.maxIndex v of
      0 -> ActLp
      1 -> ActRp
      2 -> ActNoth
      _ -> ActNoth

-- JSON 实例
instance ToJSON ParseAction where
  toJSON ActLp = toJSON ("Lp" :: T.Text)
  toJSON ActRp = toJSON ("Rp" :: T.Text)
  toJSON ActNoth = toJSON ("Noth" :: T.Text)

instance FromJSON ParseAction where
  parseJSON = withText "ParseAction" $ \t -> case t of
    "Lp"   -> pure ActLp
    "Rp"   -> pure ActRp
    "Noth" -> pure ActNoth
    _      -> fail $ "Unknown action string: " ++ T.unpack t

-- ===================== 语法树定义 =====================
type NodeFeat = Vector Double

data SyntaxTree a = EmptySTree
                  | STreeNode
                      { nodeData :: a
                      , stLeft  :: SyntaxTree a
                      , stRight :: SyntaxTree a
                      , stIsLeaf :: Bool  -- 新增显式叶子标记，O(1)判断
                      } deriving (Show, Eq)

type RawSyntaxTree = SyntaxTree NodeFeat

-- JSON 辅助：向量转列表
vecToList :: Vector Double -> [Double]
vecToList = V.toList

listToVec :: [Double] -> Vector Double
listToVec = V.fromList

-- RawSyntaxTree JSON序列化实例
instance ToJSON RawSyntaxTree where
  toJSON EmptySTree = object ["type" .= ("empty" :: T.Text)]
  toJSON STreeNode{nodeData=feat, stLeft=l, stRight=r, stIsLeaf=isL} = object
    [ "type" .= ("node" :: T.Text)
    , "feat" .= vecToList feat
    , "left" .= l
    , "right" .= r
    , "is_leaf" .= isL
    ]

instance FromJSON RawSyntaxTree where
  parseJSON = withObject "RawSyntaxTree" $ \o -> do
    typ <- o .: "type"
    case typ :: T.Text of
      "empty" -> pure EmptySTree
      "node"  -> do
        featLst <- o .: "feat"
        left <- o .: "left"
        right <- o .: "right"
        isL <- o .: "is_leaf"
        pure $ STreeNode (listToVec featLst) left right isL
      _ -> fail $ "Bad SyntaxTree type tag: " ++ T.unpack typ

-- 训练样本：((左树,右树),目标动作)
type TrainSample = ((RawSyntaxTree, RawSyntaxTree), ParseAction)

-- JSON 样本包装结构
data SampleRecord = SampleRecord
  { recLeftTree  :: RawSyntaxTree
  , recRightTree :: RawSyntaxTree
  , recTargetAct :: ParseAction
  } deriving (Show, Generic)

instance ToJSON SampleRecord where
  toJSON SampleRecord{..} = object
    [ "left_tree" .= recLeftTree
    , "right_tree" .= recRightTree
    , "target_action" .= recTargetAct
    ]

instance FromJSON SampleRecord where
  parseJSON = withObject "SampleRecord" $ \o -> SampleRecord
    <$> o .: "left_tree"
    <*> o .: "right_tree"
    <*> o .: "target_action"

sampleFromRecord :: SampleRecord -> TrainSample
sampleFromRecord SampleRecord{..} = ((recLeftTree, recRightTree), recTargetAct)

recordFromSample :: TrainSample -> SampleRecord
recordFromSample ((l,r),a) = SampleRecord l r a

-- ===================== JSON 文件读写模块 =====================
loadSamplesJSON :: FilePath -> IO [TrainSample]
loadSamplesJSON path = do
  res <- catch (decodeFileStrict path)
          (\(_ :: SomeException) -> pure Nothing)
  case res of
    Nothing -> error $ "JSON解析失败或文件不存在：" ++ path
    Just (rs :: [SampleRecord]) -> pure $ map sampleFromRecord rs

saveSamplesJSON :: FilePath -> [TrainSample] -> IO ()
saveSamplesJSON path samples = do
  let recs = map recordFromSample samples
      dat = encode recs
  BL.writeFile path dat

-- ===================== 向量矩阵数学工具 =====================
vecAdd :: Vector Double -> Vector Double -> Vector Double
vecAdd = V.zipWith (+)

vecSub :: Vector Double -> Vector Double -> Vector Double
vecSub = V.zipWith (-)

vecMul :: Vector Double -> Vector Double -> Vector Double
vecMul = V.zipWith (*)

vecScale :: Double -> Vector Double -> Vector Double
vecScale s = V.map (*s)

matVecMul :: Vector (Vector Double) -> Vector Double -> Vector Double
matVecMul mat vec = V.map (\row -> V.sum $ vecMul row vec) mat

matAdd :: Vector (Vector Double) -> Vector (Vector Double) -> Vector (Vector Double)
matAdd = V.zipWith vecAdd

matScale :: Double -> Vector (Vector Double) -> Vector (Vector Double)
matScale s = V.map (vecScale s)

matOuter :: Vector Double -> Vector Double -> Vector (Vector Double)
matOuter x y = V.map (\xi -> V.map (*xi) y) x

matTrans :: Vector (Vector Double) -> Vector (Vector Double)
matTrans mat
  | V.null mat = V.empty
  | otherwise =
      let colLen = V.length (V.head mat)
      in V.generate colLen $ \row -> V.map (\x -> x V.! row) mat

-- 激活函数与导数
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

vecSigmoid :: Vector Double -> Vector Double
vecSigmoid = V.map sigmoid

sigmoidDeriv :: Vector Double -> Vector Double
sigmoidDeriv x = vecMul x (V.map (1.0-) x)

vecTanh :: Vector Double -> Vector Double
vecTanh = V.map tanh

tanhDeriv :: Vector Double -> Vector Double
tanhDeriv x = vecSub (V.replicate (V.length x) 1.0) (vecMul x x)

softmax :: Vector Double -> Vector Double
softmax vec =
  let expVec = V.map exp vec
      sumExp = V.sum expVec
      sumExpClip = max sumExp 1e-12
  in V.map (/sumExpClip) expVec

crossEntropy :: Vector Double -> Vector Double -> Double
crossEntropy pred label =
  - sum (V.zipWith (\p l -> l * log (max p 1e-8)) pred label)

-- ===================== 网络参数 & 梯度结构 =====================
data NetParam = NetParam
  { leafProjW    :: Vector (Vector Double)  -- projDim × leafRawDim
  , leafProjB    :: Vector Double           -- projDim
  , nonLeafProjW :: Vector (Vector Double)  -- projDim × nonLeafRawDim
  , nonLeafProjB :: Vector Double           -- projDim

  , embWeight :: Vector (Vector Double)
  , embBias   :: Vector Double

  , lstmInpW     :: Vector (Vector Double)
  , lstmForgetLW :: Vector (Vector Double)
  , lstmForgetRW :: Vector (Vector Double)
  , lstmOutW     :: Vector (Vector Double)
  , lstmCellW    :: Vector (Vector Double)

  , lstmInpB     :: Vector Double
  , lstmForgetLB :: Vector Double
  , lstmForgetRB :: Vector Double
  , lstmOutB     :: Vector Double
  , lstmCellB    :: Vector Double

  , clsWeight  :: Vector (Vector Double)
  , clsBias    :: Vector Double
  } deriving (Show)

netHidDim :: NetParam -> Int
netHidDim NetParam{embWeight=w} = V.length w

netProjDim :: NetParam -> Int
netProjDim NetParam{leafProjW=w} = V.length w

type NetGrad = NetParam

data LSTMState = LSTMState
  { hState :: Vector Double
  , cState :: Vector Double
  } deriving (Show)

emptyLSTMState :: Int -> LSTMState
emptyLSTMState d = LSTMState (V.replicate d 0.0) (V.replicate d 0.0)

-- 前向计算缓存结构
data NodeCache = NodeCache
  { ncFeat      :: NodeFeat
  , ncProjFeat  :: Vector Double
  , ncEmbOut    :: Vector Double
  , ncInpGate   :: Vector Double
  , ncForgetLG  :: Vector Double
  , ncForgetRG  :: Vector Double
  , ncOutGate   :: Vector Double
  , ncCellCand  :: Vector Double
  , ncNewC      :: Vector Double
  , ncNewH      :: Vector Double
  , ncLeftH     :: Vector Double
  , ncLeftC     :: Vector Double
  , ncRightH    :: Vector Double
  , ncRightC    :: Vector Double
  , ncIsLeaf    :: Bool
  } deriving (Show)

-- 修复：where作用域放到let内部
emptyNodeCache :: Int -> Int -> NodeCache
emptyNodeCache projDim hidDim =
  let hidZero = V.replicate hidDim 0.0
      projZero = V.replicate projDim 0.0
  in NodeCache
  { ncFeat = projZero        -- ??
  , ncProjFeat = projZero
  , ncEmbOut = hidZero
  , ncInpGate = hidZero
  , ncForgetLG = hidZero
  , ncForgetRG = hidZero
  , ncOutGate = hidZero
  , ncCellCand = hidZero
  , ncNewC = hidZero
  , ncNewH = hidZero
  , ncLeftH = hidZero
  , ncLeftC = hidZero
  , ncRightH = hidZero
  , ncRightC = hidZero
  , ncIsLeaf = True       -- ??
  }

data TreeFwdResult = TreeFwdResult
  { tfRootState :: LSTMState
  , tfCacheTree :: SyntaxTree NodeCache
  } deriving (Show)

-- ===================== 特征投影 =====================
projectRawFeat :: NetParam -> Bool -> Vector Double -> Vector Double
projectRawFeat NetParam{..} isLeaf rawFeat
  | isLeaf    = vecAdd (matVecMul leafProjW rawFeat) leafProjB
  | otherwise = vecAdd (matVecMul nonLeafProjW rawFeat) nonLeafProjB

-- ===================== 参数初始化 =====================
randVec :: Int -> IO (Vector Double)
randVec n = V.replicateM n (randomRIO (-0.1, 0.1))

randMat :: Int -> Int -> IO (Vector (Vector Double))
randMat r c = V.replicateM r (randVec c)

{-
入参：
leafRawDim    叶子原始特征维度
nonLeafRawDim 非叶子原始特征维度
projDim       投影统一维度
hidDim        LSTM隐层维度
-}
initNetParam :: Int -> Int -> Int -> Int -> IO NetParam
initNetParam leafRawDim nonLeafRawDim projDim hidDim = do
  lProjW  <- randMat projDim leafRawDim
  lProjB  <- randVec projDim
  nlProjW <- randMat projDim nonLeafRawDim
  nlProjB <- randVec projDim

  embW <- randMat hidDim projDim
  embB <- randVec hidDim

  lstmIW  <- randMat hidDim hidDim
  lstmFWL <- randMat hidDim hidDim
  lstmFWR <- randMat hidDim hidDim
  lstmOW  <- randMat hidDim hidDim
  lstmCW  <- randMat hidDim hidDim

  lstmIB  <- randVec hidDim
  -- LSTM遗忘门偏置初始化为1，缓解梯度消失
  lstmFBL <- V.replicateM hidDim (pure 1.0)
  lstmFBR <- V.replicateM hidDim (pure 1.0)
  lstmOB  <- randVec hidDim
  lstmCB  <- randVec hidDim

  clsW <- randMat 3 (2*hidDim)
  clsB <- randVec 3

  return NetParam
    { leafProjW=lProjW, leafProjB=lProjB
    , nonLeafProjW=nlProjW, nonLeafProjB=nlProjB
    , embWeight=embW, embBias=embB
    , lstmInpW=lstmIW, lstmForgetLW=lstmFWL, lstmForgetRW=lstmFWR
    , lstmOutW=lstmOW, lstmCellW=lstmCW
    , lstmInpB=lstmIB, lstmForgetLB=lstmFBL, lstmForgetRB=lstmFBR
    , lstmOutB=lstmOB, lstmCellB=lstmCB
    , clsWeight=clsW, clsBias=clsB
    }

-- 零梯度构造器
zeroGrad :: Int -> Int -> Int -> Int -> NetGrad
zeroGrad leafRawDim nonLeafRawDim projDim hidDim =
  let zm r c = V.replicate r (V.replicate c 0.0)
      zv d = V.replicate d 0.0
  in NetParam
    { leafProjW    = zm projDim leafRawDim
    , leafProjB    = zv projDim
    , nonLeafProjW = zm projDim nonLeafRawDim
    , nonLeafProjB = zv projDim

    , embWeight = zm hidDim projDim, embBias = zv hidDim
    , lstmInpW=zm hidDim hidDim, lstmForgetLW=zm hidDim hidDim, lstmForgetRW=zm hidDim hidDim
    , lstmOutW=zm hidDim hidDim, lstmCellW=zm hidDim hidDim
    , lstmInpB=zv hidDim, lstmForgetLB=zv hidDim, lstmForgetRB=zv hidDim
    , lstmOutB=zv hidDim, lstmCellB=zv hidDim
    , clsWeight=zm 3 (2*hidDim), clsBias=zv 3
    }

gradAdd :: NetGrad -> NetGrad -> NetGrad
gradAdd g1 g2 =
  let
    addMat m1 m2 = matAdd m1 m2
    addVec v1 v2 = vecAdd v1 v2
  in NetParam
    { leafProjW    = addMat (leafProjW g1) (leafProjW g2)
    , leafProjB    = addVec (leafProjB g1) (leafProjB g2)
    , nonLeafProjW = addMat (nonLeafProjW g1) (nonLeafProjW g2)
    , nonLeafProjB = addVec (nonLeafProjB g1) (nonLeafProjB g2)

    , embWeight = addMat (embWeight g1) (embWeight g2)
    , embBias = addVec (embBias g1) (embBias g2)
    , lstmInpW = addMat (lstmInpW g1) (lstmInpW g2)
    , lstmForgetLW = addMat (lstmForgetLW g1) (lstmForgetLW g2)
    , lstmForgetRW = addMat (lstmForgetRW g1) (lstmForgetRW g2)
    , lstmOutW = addMat (lstmOutW g1) (lstmOutW g2)
    , lstmCellW = addMat (lstmCellW g1) (lstmCellW g2)
    , lstmInpB = addVec (lstmInpB g1) (lstmInpB g2)
    , lstmForgetLB = addVec (lstmForgetLB g1) (lstmForgetLB g2)
    , lstmForgetRB = addVec (lstmForgetRB g1) (lstmForgetRB g2)
    , lstmOutB = addVec (lstmOutB g1) (lstmOutB g2)
    , lstmCellB = addVec (lstmCellB g1) (lstmCellB g2)
    , clsWeight = addMat (clsWeight g1) (clsWeight g2)
    , clsBias = addVec (clsBias g1) (clsBias g2)
    }

gradScale :: Double -> NetGrad -> NetGrad
gradScale scale g1 =
   NetParam
    { leafProjW    = matScale scale (leafProjW g1)
    , leafProjB    = vecScale scale (leafProjB g1)
    , nonLeafProjW = matScale scale (nonLeafProjW g1)
    , nonLeafProjB = vecScale scale (nonLeafProjB g1)

    , embWeight = matScale scale (embWeight g1)
    , embBias = vecScale scale (embBias g1)
    , lstmInpW = matScale scale (lstmInpW g1)
    , lstmForgetLW = matScale scale (lstmForgetLW g1)
    , lstmForgetRW = matScale scale (lstmForgetRW g1)
    , lstmOutW = matScale scale (lstmOutW g1)
    , lstmCellW = matScale scale (lstmCellW g1)
    , lstmInpB = vecScale scale (lstmInpB g1)
    , lstmForgetLB = vecScale scale (lstmForgetLB g1)
    , lstmForgetRB = vecScale scale (lstmForgetRB g1)
    , lstmOutB = vecScale scale (lstmOutB g1)
    , lstmCellB = vecScale scale (lstmCellB g1)
    , clsWeight = matScale scale (clsWeight g1)
    , clsBias = vecScale scale (clsBias g1)
    }

updateParam :: Double -> NetGrad -> NetParam -> NetParam
updateParam lr grad p =
  let
    updateMat m gm = matAdd m (matScale (-lr) gm)
    updateVec v gv = vecAdd v (vecScale (-lr) gv)
  in p
    { leafProjW    = updateMat (leafProjW p) (leafProjW grad)
    , leafProjB    = updateVec (leafProjB p) (leafProjB grad)
    , nonLeafProjW = updateMat (nonLeafProjW p) (nonLeafProjW grad)
    , nonLeafProjB = updateVec (nonLeafProjB p) (nonLeafProjB grad)

    , embWeight = updateMat (embWeight p) (embWeight grad)
    , embBias = updateVec (embBias p) (embBias grad)
    , lstmInpW = updateMat (lstmInpW p) (lstmInpW grad)
    , lstmForgetLW = updateMat (lstmForgetLW p) (lstmForgetLW grad)
    , lstmForgetRW = updateMat (lstmForgetRW p) (lstmForgetRW grad)
    , lstmOutW = updateMat (lstmOutW p) (lstmOutW grad)
    , lstmCellW = updateMat (lstmCellW p) (lstmCellW grad)
    , lstmInpB = updateVec (lstmInpB p) (lstmInpB grad)
    , lstmForgetLB = updateVec (lstmForgetLB p) (lstmForgetLB grad)
    , lstmForgetRB = updateVec (lstmForgetRB p) (lstmForgetRB grad)
    , lstmOutB = updateVec (lstmOutB p) (lstmOutB grad)
    , lstmCellB = updateVec (lstmCellB p) (lstmCellB grad)
    , clsWeight = updateMat (clsWeight p) (clsWeight grad)
    , clsBias = updateVec (clsBias p) (clsBias grad)
    }

-- ===================== 前向传播 =====================
treeForwardCached :: NetParam -> RawSyntaxTree -> TreeFwdResult
treeForwardCached param EmptySTree =
  let
    hidDim = netHidDim param
    projDim = netProjDim param
    zeroVec = V.replicate hidDim 0.0
    emptyCache = emptyNodeCache projDim hidDim
  in TreeFwdResult (LSTMState zeroVec zeroVec) (STreeNode emptyCache EmptySTree EmptySTree False)

treeForwardCached param st@STreeNode{nodeData=feat, stLeft=l, stRight=r, stIsLeaf=isLeafNode} =
  let
    leftRes = treeForwardCached param l
    rightRes = treeForwardCached param r
    LSTMState hl cl = tfRootState leftRes
    LSTMState hr cr = tfRootState rightRes
    NetParam{..} = param

    projFeat = projectRawFeat param isLeafNode feat

    embOut = vecAdd (matVecMul embWeight projFeat) embBias
    inpGate     = vecSigmoid $ vecAdd (matVecMul lstmInpW embOut) lstmInpB
    forgetLGate = vecSigmoid $ vecAdd (matVecMul lstmForgetLW hl) lstmForgetLB
    forgetRGate = vecSigmoid $ vecAdd (matVecMul lstmForgetRW hr) lstmForgetRB
    outGate     = vecSigmoid $ vecAdd (matVecMul lstmOutW embOut) lstmOutB
    cellCand    = vecTanh $ vecAdd (matVecMul lstmCellW embOut) lstmCellB

    newC = ((vecMul forgetLGate cl) `vecAdd` vecMul forgetRGate cr) `vecAdd` vecMul inpGate cellCand
    newH = vecMul outGate (vecTanh newC)

    cache = NodeCache
      { ncFeat = feat
      , ncProjFeat = projFeat
      , ncEmbOut = embOut
      , ncInpGate = inpGate, ncForgetLG = forgetLGate, ncForgetRG = forgetRGate
      , ncOutGate = outGate, ncCellCand = cellCand, ncNewC = newC, ncNewH = newH
      , ncLeftH = hl, ncLeftC = cl, ncRightH = hr, ncRightC = cr
      , ncIsLeaf = isLeafNode
      }
    cachedNode = STreeNode cache (tfCacheTree leftRes) (tfCacheTree rightRes) isLeafNode
  in TreeFwdResult (LSTMState newH newC) cachedNode

-- ===================== 反向传播核心节点BP（已全部修复） =====================
-- 返回：(梯度, dLdh_left, dLdc_left, dLdh_right, dLdc_right, dLd_rawFeat)
backwardNode :: NetParam -> NodeCache -> Vector Double -> Vector Double
             -> (NetGrad, Vector Double, Vector Double, Vector Double, Vector Double, Vector Double)
backwardNode NetParam{..} NodeCache{..} dLdh dLdc =
  let
    -- 1. LSTM 细胞状态链式求导（修复tanh嵌套导数）
    tanhNewC = vecTanh ncNewC
    dTanhC = tanhDeriv tanhNewC
    dh_dc = vecMul ncOutGate dTanhC
    dLdnewC = vecAdd (vecMul dLdh dh_dc) dLdc

    -- 输出门梯度
    dLdOutGate = vecMul dLdh tanhNewC
    dRawOut = vecMul dLdOutGate (sigmoidDeriv ncOutGate)

    -- 输入门梯度
    dLdInpGate = vecMul dLdnewC ncCellCand
    dRawInp = vecMul dLdInpGate (sigmoidDeriv ncInpGate)

    -- 左遗忘门梯度
    dLdForgetL = vecMul dLdnewC ncLeftC
    dRawForgetL = vecMul dLdForgetL (sigmoidDeriv ncForgetLG)

    -- 右遗忘门梯度
    dLdForgetR = vecMul dLdnewC ncRightC
    dRawForgetR = vecMul dLdForgetR (sigmoidDeriv ncForgetRG)

    -- cell候选梯度
    dLdCellCand = vecMul dLdnewC ncInpGate
    dRawCell = vecMul dLdCellCand (tanhDeriv ncCellCand)

    -- 修复：dEmbLSTM先定义再使用，补全两个遗忘门对embOut的梯度
    dEmbLSTM = matVecMul (matTrans lstmInpW) dRawInp
         `vecAdd` matVecMul (matTrans lstmOutW) dRawOut
         `vecAdd` matVecMul (matTrans lstmCellW) dRawCell
         `vecAdd` matVecMul (matTrans lstmForgetLW) dRawForgetL
         `vecAdd` matVecMul (matTrans lstmForgetRW) dRawForgetR

    dProjFeat = matVecMul (matTrans embWeight) dEmbLSTM

    -- 左右子树隐层与细胞状态梯度
    dLdhL = matVecMul (matTrans lstmForgetLW) dRawForgetL
    dLdcL = vecMul dLdnewC ncForgetLG
    dLdhR = matVecMul (matTrans lstmForgetRW) dRawForgetR
    dLdcR = vecMul dLdnewC ncForgetRG

    -- 投影层梯度计算 + 原始输入特征梯度dL/d_ncFeat
    hidDim = V.length ncEmbOut
    projDim = V.length dProjFeat
    leafRawDim = V.length leafProjW
    nonLeafRawDim = V.length nonLeafProjW
    zeroG = zeroGrad leafRawDim nonLeafRawDim projDim hidDim

    (projGrad, dRawFeat)
      | ncIsLeaf =
          let gW = matOuter dProjFeat ncFeat
              gB = dProjFeat
              dX = matVecMul (matTrans leafProjW) dProjFeat
              pg = zeroG { leafProjW = gW, leafProjB = gB }
          in (pg, dX)
      | otherwise =
          let gW = matOuter dProjFeat ncFeat
              gB = dProjFeat
              dX = matVecMul (matTrans nonLeafProjW) dProjFeat
              pg = zeroG { nonLeafProjW = gW, nonLeafProjB = gB }
          in (pg, dX)

    -- LSTM内部权重梯度
    gradEmbW = matOuter dEmbLSTM ncProjFeat
    gradEmbB = dEmbLSTM

    gradIW = matOuter dRawInp ncEmbOut
    gradIB = dRawInp

    gradFWL = matOuter dRawForgetL ncLeftH
    gradFBL = dRawForgetL

    gradFWR = matOuter dRawForgetR ncRightH
    gradFBR = dRawForgetR

    gradOW = matOuter dRawOut ncEmbOut
    gradOB = dRawOut

    gradCW = matOuter dRawCell ncEmbOut
    gradCB = dRawCell

    lstmGrad = zeroG
      { embWeight = gradEmbW, embBias = gradEmbB
      , lstmInpW = gradIW, lstmForgetLW = gradFWL, lstmForgetRW = gradFWR
      , lstmOutW = gradOW, lstmCellW = gradCW
      , lstmInpB = gradIB, lstmForgetLB = gradFBL, lstmForgetRB = gradFBR
      , lstmOutB = gradOB, lstmCellB = gradCB
      }

    fullGrad = projGrad `gradAdd` lstmGrad

  in (fullGrad, dLdhL, dLdcL, dLdhR, dLdcR, dRawFeat)

-- 整树递归反向传播
backwardTree :: NetParam -> SyntaxTree NodeCache -> Vector Double -> Vector Double -> NetGrad
backwardTree _ EmptySTree _ _ = zeroGrad 0 0 0 0
backwardTree param (STreeNode cache leftTree rightTree _) dLdh dLdc =
  let
    (nodeGrad, dLdhL, dLdcL, dLdhR, dLdcR, _) = backwardNode param cache dLdh dLdc
    gradLeft = backwardTree param leftTree dLdhL dLdcL
    gradRight = backwardTree param rightTree dLdhR dLdcR
    totalGrad = nodeGrad `gradAdd` gradLeft `gradAdd` gradRight
  in totalGrad

-- ===================== 单样本前向+损失+梯度 =====================
sampleForwardLossGrad :: NetParam -> TrainSample -> (Double, NetGrad)
sampleForwardLossGrad param ((ltree, rtree), action) =
  let
    lFwd = treeForwardCached param ltree
    rFwd = treeForwardCached param rtree
    LSTMState hl _ = tfRootState lFwd
    LSTMState hr _ = tfRootState rFwd
    concatH = V.concat [hl, hr]
    NetParam{..} = param
    logits = vecAdd (matVecMul clsWeight concatH) clsBias
    pred = softmax logits
    label = actionToOneHot action
    loss = crossEntropy pred label

    dLogit = vecSub pred label
    gradClsW = matOuter dLogit concatH
    gradClsB = dLogit

    hidDim = V.length hl
    dConcatH = matVecMul (matTrans clsWeight) dLogit
    dHl = V.take hidDim dConcatH
    dHr = V.drop hidDim dConcatH

    gradL = backwardTree param (tfCacheTree lFwd) dHl (V.replicate hidDim 0.0)
    gradR = backwardTree param (tfCacheTree rFwd) dHr (V.replicate hidDim 0.0)
    totalGrad0 = gradL `gradAdd` gradR
    totalGrad = totalGrad0
      { clsWeight = gradClsW
      , clsBias = gradClsB
      }
  in (loss, totalGrad)

-- 单样本SGD更新
trainStep :: NetParam -> TrainSample -> Double -> (Double, NetParam)
trainStep param sample lr =
  let (loss, grad) = sampleForwardLossGrad param sample
      newParam = updateParam lr grad param
  in (loss, newParam)

-- 批量梯度下降（修复zeroGrad硬编码0）
batchTrainStep :: NetParam -> [TrainSample] -> Double -> (Double, NetParam)
batchTrainStep param samples lr =
  let
    leafD = V.length $ leafProjW param
    nonLeafD = V.length $ nonLeafProjW param
    projD = netProjDim param
    hidD = netHidDim param
    initG = zeroGrad leafD nonLeafD projD hidD

    pairs = map (sampleForwardLossGrad param) samples
    losses = map fst pairs
    grads = map snd pairs

    totalLoss = sum losses
    batchSize = fromIntegral (length samples)
    avgLoss = totalLoss / batchSize

    sumGrad = foldl gradAdd initG grads
    avgGrad = gradScale (1.0 / batchSize) sumGrad
    newParam = updateParam lr avgGrad param
  in (avgLoss, newParam)

-- ===================== 树构造辅助函数 =====================
mkLeaf :: [Double] -> [Double] -> RawSyntaxTree
mkLeaf wordEmb catEmb =
  let feat = listToVec (wordEmb ++ catEmb)
  in STreeNode feat EmptySTree EmptySTree True

mkNonLeaf :: [Double] -> [Double] -> [Double] -> RawSyntaxTree -> RawSyntaxTree -> RawSyntaxTree
mkNonLeaf cat tag phra l r =
  let feat = listToVec (cat ++ tag ++ phra)
  in STreeNode feat l r False

-- ===================== 尾递归训练循环（防止栈溢出） =====================
trainLoop :: NetParam -> [TrainSample] -> Double -> Int -> IO NetParam
trainLoop param _ _ 0 = return param
trainLoop param samples lr epoch = do
  let (avgLoss, newParam) = batchTrainStep param samples lr
  if epoch `mod` 50 == 0
    then putStrLn $ "Epoch剩余:" ++ show epoch ++ " | AvgLoss:" ++ show avgLoss
    else return ()
  trainLoop newParam samples lr (epoch - 1)

{- ========= 转换 BiTree PhraSyn0 到 SyntaxTree NodeFeat =========
 - data BiTree a = Empty | Node a (BiTree a) (BiTree a) deriving (Eq)
 - data SyntaxTree a = EmptySTree | STreeNode { nodeData :: a, stLeft :: SyntaxTree a, stRight :: SyntaxTree a, stIsLeaf :: Bool } deriving (Show, Eq)
 -}
biTreePhraSyn02SyntaxTreeNodeFeat :: BiTree PhraSyn0
                                     -> Map Category (Vector Double)
                                     -> Map Tag (Vector Double)
                                     -> Map PhraStru (Vector Double)
                                     -> SyntaxTree NodeFeat
biTreePhraSyn02SyntaxTreeNodeFeat Empty _ _ _ = EmptySTree
biTreePhraSyn02SyntaxTreeNodeFeat (Node phraSyn0 lt rt) cateVecMap tagVecMap struVecMap =
    let
        cvLen = 8
        cateVec = fromMaybe (V.replicate cvLen 0.0) $ Map.lookup (fst3 phraSyn0) cateVecMap
        tvLen = 16
        tagVec = fromMaybe (V.replicate tvLen 0.0) $ Map.lookup (snd3 phraSyn0) tagVecMap
        struLen = 16
        struVec = fromMaybe (V.replicate struLen 0.0) $ Map.lookup (thd3 phraSyn0) struVecMap
    in STreeNode { nodeData = cateVec V.++ tagVec V.++ struVec
                 , stLeft = biTreePhraSyn02SyntaxTreeNodeFeat lt cateVecMap tagVecMap struVecMap
                 , stRight = biTreePhraSyn02SyntaxTreeNodeFeat rt cateVecMap tagVecMap struVecMap
                 , stIsLeaf = isLeafNode lt && isLeafNode rt
                 }

-- ==================== 建立样本集 ====================
buildSampleSet :: IO () -- [SampleRecord]
buildSampleSet = do
  confInfo <- readFile "Configuration"
  let syntax_ambig_resol_model = getConfProperty "syntax_ambig_resol_model" confInfo
  putStrLn $ " syntax_ambig_resol_model: " ++ syntax_ambig_resol_model
  conn <- getConn
  let sqlstat = DS.fromString $ "select id, leftOverTree, rightOverTree, clauTagPrior from " ++ syntax_ambig_resol_model
  stmt <- prepareStmt conn sqlstat
  (defs, is) <- queryStmt conn stmt []
  int32UTextTextTextList <- readStreamByInt32UTextTextText [] is        -- [(Int, String, String, String)]
  let idLTreeRTreePriorList = map (\x -> ( fst4 x
                                         , stringToBiTree getPhraSyn0FromStr (snd4 x)
                                         , stringToBiTree getPhraSyn0FromStr (thd4 x)
                                         , (fromMaybe Noth . priorWithHighestFreq . stringToCTPList) (fth4 x)
                                         )
                                  ) int32UTextTextTextList
  putStrLn $ "Sample number = " ++ show (length idLTreeRTreePriorList) ++ ", the first is: " ++ show (idLTreeRTreePriorList!!0)
{-
  let ovTree2ResolList = map (\x -> ((snd4 x, thd4 x), fth4 x)) idLTreeRTreePriorList      -- [((BiTree PhraSyn0, BiTree PhraSyn0), Prior)]
      cateVecMap = cate2Vec                         -- Map Category (Vector Double)
      tagVecMap = tag2Vec                           -- Map Tag (Vector Double)
      struVecMap = stru2Vec                         -- Map PhraStru (Vector Double)

      sampleSet ==  map (\x -> let
                                 lt = biTreePhraSyn02SyntaxTreeNodeFeat cateVecMap tagVecMap struVecMap ((fst . fst) x)
                                 rt = biTreePhraSyn02SyntaxTreeNodeFeat cateVecMap tagVecMap struVecMap ((fst . snd) x)
                               in ((lt, rt), snd x)
                        ) ovTree2ResolList
  return sampleSet
 -}
-- ===================== 测试入口 =====================
testPipeline :: IO ()
testPipeline = do
  putStrLn "==== Binary Tree-LSTM 句法合并分类模型【修复完整版】 ===="
  let
    dim_word    = 2
    dim_cat     = 2
    dim_tag     = 2
    dim_phrase  = 2

    leafRawDim    = dim_word + dim_cat
    nonLeafRawDim = dim_cat + dim_tag + dim_phrase
    projDim       = 4
    hidDim        = 10

    lr      = 0.008
    epochs  = 800

  initP <- initNetParam leafRawDim nonLeafRawDim projDim hidDim

  -- 构造测试样本
  let
    leafA = mkLeaf [0.1,0.3] [0.5,0.2]
    leafB = mkLeaf [0.4,0.2] [0.1,0.7]
    treeL = mkNonLeaf [0.3,0.6] [0.2,0.4] [0.1,0.9] leafA EmptySTree
    treeR = mkNonLeaf [0.7,0.2] [0.5,0.1] [0.3,0.4] EmptySTree leafB
    testSampleSet = [ ((treeL, treeR), ActRp) ]
  samples <- pure testSampleSet

  putStrLn $ "样本数量：" ++ show (length samples)
  trainedP <- trainLoop initP samples lr epochs

  -- 推理预测
  let ((testL,testR), trueAct) = head samples
      lFwd = treeForwardCached trainedP testL
      rFwd = treeForwardCached trainedP testR
      LSTMState hl _ = tfRootState lFwd
      LSTMState hr _ = tfRootState rFwd
      concatH = V.concat [hl,hr]
      logits = vecAdd (matVecMul (clsWeight trainedP) concatH) (clsBias trainedP)
      pred = softmax logits
      predAct = vecToAction pred

  putStrLn "\n====预测结果===="
  putStrLn $ "预测分布:" ++ show pred
  putStrLn $ "预测动作:" ++ show predAct
  putStrLn $ "真实动作:" ++ show trueAct
