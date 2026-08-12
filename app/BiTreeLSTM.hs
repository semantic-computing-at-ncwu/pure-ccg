{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

-- Binary Tree-LSTM 句法合并分类模型
-- 任务：输入一对语法树(LeftTree, RightTree)，预测 ActLp / ActRp / ActNoth 三分类
-- 完整前向 + 递归反向传播BP + JSON样本读写
module BiTreeLSTM where

import qualified Data.Vector as V
import Data.Vector (Vector)
import System.Random (randomRIO)
import Data.Foldable (sum)
import GHC.Generics (Generic)
import Control.Exception (catch, SomeException)
import System.IO (readFile, writeFile)

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
vecToAction v = case V.maxIndex v of
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

-- ===================== 语法树定义：只保留泛型 SyntaxTree a =====================
type NodeFeat = Vector Double

data SyntaxTree a = EmptySTree
                  | STreeNode
                      { nodeData :: a
                      , stLeft  :: SyntaxTree a
                      , stRight :: SyntaxTree a
                      } deriving (Show)

-- 原始输入树：不带缓存，承载 NodeFeat
type RawSyntaxTree = SyntaxTree NodeFeat

-- JSON 辅助：向量转列表
vecToList :: Vector Double -> [Double]
vecToList = V.toList

listToVec :: [Double] -> Vector Double
listToVec = V.fromList

-- RawSyntaxTree JSON序列化实例
instance ToJSON RawSyntaxTree where
  toJSON EmptySTree = object ["type" .= ("empty" :: T.Text)]
  toJSON STreeNode{nodeData=feat, stLeft=l, stRight=r} = object
    [ "type" .= ("node" :: T.Text)
    , "feat" .= vecToList feat
    , "left" .= l
    , "right" .= r
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
        pure $ STreeNode (listToVec featLst) left right
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
-- 加载JSON样本文件
loadSamplesJSON :: FilePath -> IO [TrainSample]
loadSamplesJSON path = do
  mbs <- decodeFileStrict path
  case mbs of
    Nothing -> error $ "JSON解析失败：" ++ path
    Just (rs :: [SampleRecord]) -> pure $ map sampleFromRecord rs

-- 保存样本到JSON文件
saveSamplesJSON :: FilePath -> [TrainSample] -> IO ()
saveSamplesJSON path samples = do
  let recs = map recordFromSample samples
      dat = encode recs
  BL.writeFile path dat

-- ===================== 向量数学工具 =====================
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
matTrans mat =
    let
      colLen = V.length (head mat)
    in V.generate colLen $ \row -> V.map (! row) mat

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
  in V.map (/sumExp) expVec

{- 'label' are real probabilities of all events.
 - 'pred' are predictive probabilities of all events.
 -}
crossEntropy :: Vector Double -> Vector Double -> Double
crossEntropy pred label = - sum (V.zipWith (\p l -> l * log (p + 1e-8)) pred label)

-- ===================== 网络参数 & 梯度结构 =====================
data NetParam = NetParam
  { embWeight :: Vector (Vector Double)
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

-- 梯度与参数同构
type NetGrad = NetParam

data LSTMState = LSTMState
  { hState :: Vector Double
  , cState :: Vector Double
  } deriving (Show)

-- 'd' is the dimension, that is, the number of LSTM basic cells.
emptyLSTMState :: Int -> LSTMState
emptyLSTMState d = LSTMState (V.replicate d 0.0) (V.replicate d 0.0)

-- 前向缓存：单个节点所有中间变量，用于反向传播
data NodeCache = NodeCache
  { ncFeat      :: NodeFeat
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
  } deriving (Show)

-- 整树前向同时返回缓存（用于BP）
data TreeFwdResult = TreeFwdResult
  { tfRootState :: LSTMState
  , tfCacheTree :: SyntaxTree NodeCache  -- 带缓存的树
  }

-- ===================== 参数初始化 =====================
randVec :: Int -> IO (Vector Double)
randVec n = V.replicateM n (randomRIO (-0.1, 0.1))

randMat :: Int -> Int -> IO (Vector (Vector Double))
randMat r c = V.replicateM r (randVec c)

{- 输入样本的嵌入式向量维度: featDim, 基于记忆单元的个数: hidDim
 - 输入权重矩阵：hidDim >< featDim
 - 输入门权重矩阵、左遗忘门权重矩阵、右遗忘门权重矩阵、输出门权重矩阵、细胞状态门权重矩阵：hidDim >< hidDim
 - 三分类神经元权重矩阵：3 >< (2 * hidDim)
 -}
initNetParam :: Int -> Int -> IO NetParam
initNetParam featDim hidDim = do
  embW <- randMat hidDim featDim
  embB <- randVec hidDim

  lstmIW  <- randMat hidDim hidDim
  lstmFWL <- randMat hidDim hidDim
  lstmFWR <- randMat hidDim hidDim
  lstmOW  <- randMat hidDim hidDim
  lstmCW  <- randMat hidDim hidDim

  lstmIB  <- randVec hidDim
  lstmFBL <- randVec hidDim
  lstmFBR <- randVec hidDim
  lstmOB  <- randVec hidDim
  lstmCB  <- randVec hidDim

  clsW <- randMat 3 (2*hidDim)
  clsB <- randVec 3

  return $ NetParam
    { embWeight=embW, embBias=embB
    , lstmInpW=lstmIW, lstmForgetLW=lstmFWL, lstmForgetRW=lstmFWR
    , lstmOutW=lstmOW, lstmCellW=lstmCW
    , lstmInpB=lstmIB, lstmForgetLB=lstmFBL, lstmForgetRB=lstmFBR
    , lstmOutB=lstmOB, lstmCellB=lstmCB
    , clsWeight=clsW, clsBias=clsB
    }

zeroGrad :: Int -> Int -> NetGrad
zeroGrad featDim hidDim =
  let zm r c = V.replicate r (V.replicate c 0.0)
      zv d = V.replicate d 0.0
  in NetParam
    { embWeight = zm hidDim featDim, embBias = zv hidDim
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
    { embWeight = addMat (embWeight g1) (embWeight g2)
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

{- Learning rate: lr
 - Original matrix or vector: m or v
 - Gradient matrix or vector: gm or gv
 - Updated matrix or vector: m - lr * gm, or v - lr * gv
 -}
updateParam :: Double -> NetGrad -> NetParam -> NetParam
updateParam lr grad p =
  let
    updateMat m gm = matAdd m (matScale (-lr) gm)
    updateVec v gv = vecAdd v (vecScale (-lr) gv)
  in p
    { embWeight = updateMat (embWeight p) (embWeight grad)
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

-- ===================== 带缓存的前向传播 =====================
treeForwardCached :: NetParam -> RawSyntaxTree -> TreeFwdResult
treeForwardCached _ EmptySTree = error "Empty tree cannot be encoded"
treeForwardCached param st@(STreeNode feat l r) =
  let
    leftRes = treeForwardCached param l
    rightRes = treeForwardCached param r
    LSTMState hl cl = tfRootState leftRes
    LSTMState hr cr = tfRootState rightRes
    NetParam{..} = param

    embOut = vecAdd (matVecMul embWeight feat) embBias                          -- W * X + B
    inpGate     = vecSigmoid $ vecAdd (matVecMul lstmInpW embOut) lstmInpB      -- Control signal of input gate
    forgetLGate = vecSigmoid $ vecAdd (matVecMul lstmForgetLW hl) lstmForgetLB  -- Control signal of left forget gate
    forgetRGate = vecSigmoid $ vecAdd (matVecMul lstmForgetRW hr) lstmForgetRB  -- Control signal of right forget gate
    outGate     = vecSigmoid $ vecAdd (matVecMul lstmOutW embOut) lstmOutB      -- Control signal of output gate
    cellCand    = vecTanh $ vecAdd (matVecMul lstmCellW embOut) lstmCellB       -- Cell state candidate 1

    newC = vecAdd (vecMul forgetLGate cl) (vecMul forgetRGate cr) `vecAdd` vecMul inpGate cellCand   -- Cell state
    newH = vecMul outGate (vecTanh newC)                                        -- Hidden state

    cache = NodeCache
      { ncFeat = feat, ncEmbOut = embOut
      , ncInpGate = inpGate, ncForgetLG = forgetLGate, ncForgetRG = forgetRGate
      , ncOutGate = outGate, ncCellCand = cellCand, ncNewC = newC, ncNewH = newH
      , ncLeftH = hl, ncLeftC = cl, ncRightH = hr, ncRightC = cr
      }
    cachedNode = STreeNode cache (tfCacheTree leftRes) (tfCacheTree rightRes)
  in TreeFwdResult (LSTMState newH newC) cachedNode

-- ===================== 完整反向传播 BP 实现 =====================
-- 返回：(梯度, dLdh_left, dLdc_left, dLdh_right, dLdc_right)
backwardNode :: NetParam -> NodeCache -> Vector Double -> Vector Double -> (NetGrad, Vector Double, Vector Double, Vector Double, Vector Double)
backwardNode NetParam{..} NodeCache{..} dLdh dLdc =
  let
    -- dL/dnewC
    dTanhC = tanhDeriv (vecTanh ncNewC)
    dLdnewC = vecAdd (vecMul dLdh ncOutGate) (vecMul dLdc dTanhC)

    -- 各门梯度
    dLdOutGate = vecMul dLdh (vecTanh ncNewC)
    dRawOut = vecMul dLdOutGate (sigmoidDeriv ncOutGate)

    dLdInpGate = vecMul dLdnewC ncCellCand
    dRawInp = vecMul dLdInpGate (sigmoidDeriv ncInpGate)

    dLdForgetL = vecMul dLdnewC ncLeftC
    dRawForgetL = vecMul dLdForgetL (sigmoidDeriv ncForgetLG)

    dLdForgetR = vecMul dLdnewC ncRightC
    dRawForgetR = vecMul dLdForgetR (sigmoidDeriv ncForgetRG)

    dLdCellCand = vecMul dLdnewC ncInpGate
    dRawCell = vecMul dLdCellCand (tanhDeriv ncCellCand)

    -- 输入emb反向
    dEmb = matVecMul (matTrans lstmInpW) dRawInp
         `vecAdd` matVecMul (matTrans lstmOutW) dRawOut
         `vecAdd` matVecMul (matTrans lstmCellW) dRawCell

    -- 左右子树dh dc
    dLdhL = matVecMul (matTrans lstmForgetLW) dRawForgetL
    dLdcL = vecMul dLdnewC ncForgetLG
    dLdhR = matVecMul (matTrans lstmForgetRW) dRawForgetR
    dLdcR = vecMul dLdnewC ncForgetRG

    -- 参数梯度
    gradEmbW = matOuter dEmb ncFeat
    gradEmbB = dEmb

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

    grad = NetParam
      { embWeight = gradEmbW, embBias = gradEmbB
      , lstmInpW = gradIW, lstmForgetLW = gradFWL, lstmForgetRW = gradFWR
      , lstmOutW = gradOW, lstmCellW = gradCW
      , lstmInpB = gradIB, lstmForgetLB = gradFBL, lstmForgetRB = gradFBR
      , lstmOutB = gradOB, lstmCellB = gradCB
      , clsWeight = V.replicate 3 (V.replicate (V.length ncEmbOut) 0.0)
      , clsBias = V.replicate 3 0.0
      }
  in (grad, dLdhL, dLdcL, dLdhR, dLdcR)

-- 递归整树反向传播
backwardTree :: NetParam -> SyntaxTree NodeCache -> Vector Double -> Vector Double -> NetGrad
backwardTree _ EmptySTree _ _ = zeroGrad 0 0
backwardTree param (STreeNode cache leftTree rightTree) dLdh dLdc =
  let
    (nodeGrad, dLdhL, dLdcL, dLdhR, dLdcR) = backwardNode param cache dLdh dLdc
    gradLeft = backwardTree param leftTree dLdhL dLdcL
    gradRight = backwardTree param rightTree dLdhR dLdcR
    totalGrad = nodeGrad `gradAdd` gradLeft `gradAdd` gradRight
  in totalGrad

-- ===================== 单样本前向+损失+完整梯度 =====================
sampleForwardLossGrad :: NetParam -> TrainSample -> (Double, NetGrad)
sampleForwardLossGrad param ((ltree, rtree), action) =
  let
    -- 两棵树前向带缓存
    lFwd@TreeFwdResult{tfRootState = LSTMState hl _} = treeForwardCached param ltree
    rFwd@TreeFwdResult{tfRootState = LSTMState hr _} = treeForwardCached param rtree
    concatH = V.concat [hl, hr]
    NetParam{..} = param
    logits = vecAdd (matVecMul clsWeight concatH) clsBias
    pred = softmax logits
    label = actionToOneHot action
    loss = crossEntropy pred label

    -- 输出层梯度 dL/dlogit
    dLogit = vecSub pred label
    -- 分类参数梯度
    gradClsW = matOuter dLogit concatH
    gradClsB = dLogit
    -- 分割梯度给左右树隐状态
    hidDim = V.length hl
    dConcatH = matVecMul (matTrans clsWeight) dLogit
    dHl = V.take hidDim dConcatH
    dHr = V.drop hidDim dConcatH

    -- 两棵树递归BP
    gradL = backwardTree param (tfCacheTree lFwd) dHl (V.replicate hidDim 0.0)
    gradR = backwardTree param (tfCacheTree rFwd) dHr (V.replicate hidDim 0.0)
    totalGrad0 = gradL `gradAdd` gradR
    totalGrad = totalGrad0
      { clsWeight = gradClsW
      , clsBias = gradClsB
      }
  in (loss, totalGrad)

-- SGD单步训练
trainStep :: NetParam -> TrainSample -> Double -> (Double, NetParam)
trainStep param sample lr =
  let (loss, grad) = sampleForwardLossGrad param sample
      newParam = updateParam lr grad param
  in (loss, newParam)

-- 简易批量梯度累加（梯度求和后更新）
batchTrainStep :: NetParam -> [TrainSample] -> Double -> (Double, NetParam)
batchTrainStep param samples lr =
  let
    pairs = map (sampleForwardLossGrad param) samples
    losses = map fst pairs
    grads = map snd pairs
    avgLoss = sum losses / fromIntegral (length samples)
    fullGrad = foldl gradAdd (zeroGrad 0 0) grads
    scaleGrad = matScale (1.0 / fromIntegral (length samples)) fullGrad
    newParam = updateParam lr scaleGrad param
  in (avgLoss, newParam)

-- ===================== 树构造辅助函数 =====================
mkLeaf :: [Double] -> [Double] -> RawSyntaxTree
mkLeaf wordEmb catEmb =
  let feat = V.fromList (wordEmb ++ catEmb)
  in STreeNode feat EmptySTree EmptySTree

mkNonLeaf :: [Double] -> [Double] -> [Double] -> RawSyntaxTree -> RawSyntaxTree -> RawSyntaxTree
mkNonLeaf cat tag phra l r =
  let feat = V.fromList (cat ++ tag ++ phra)
  in STreeNode feat l r

-- ===================== 训练循环与测试入口 =====================
trainLoop :: NetParam -> [TrainSample] -> Double -> Int -> IO NetParam
trainLoop param _ _ 0 = return param
trainLoop param samples lr epoch = do
  let (avgLoss, newParam) = batchTrainStep param samples lr
  if epoch `mod` 50 == 0
    then putStrLn $ "Epoch剩余:" ++ show epoch ++ " | AvgLoss:" ++ show avgLoss
    else return ()
  trainLoop newParam samples lr (epoch-1)

testPipeline :: IO ()
testPipeline = do
  putStrLn "==== Binary Tree-LSTM 句法合并分类模型 (完整BP + JSON加载) ===="
  let featDim = 4
      hidDim  = 10
      lr      = 0.008
      epochs  = 800

  initP <- initNetParam featDim hidDim

  -- 两种加载方式任选
  -- 1. 从JSON文件加载样本
  -- samples <- loadSamplesJSON "train_samples.json"
  -- 2. 使用内置测试样本
  let
    leafA = mkLeaf [0.1,0.3] [0.5,0.2]
    leafB = mkLeaf [0.4,0.2] [0.1,0.7]
    treeL = mkNonLeaf [0.3,0.6] [0.2,0.4] [0.1,0.9] leafA EmptySTree
    treeR = mkNonLeaf [0.7,0.2] [0.5,0.1] [0.3,0.4] EmptySTree leafB
    testSampleSet = [ ((treeL, treeR), ActRp) ]
  samples <- pure testSampleSet

  putStrLn $ "样本数量：" ++ show (length samples)
  trainedP <- trainLoop initP samples lr epochs

  -- 预测
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
