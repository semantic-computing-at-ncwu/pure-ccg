{-# LANGUAGE NamedFieldPuns, RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict #-}

-- Binary Tree-LSTM 词法消岐模型（Tai et al. 2015 Child-Sum 标准实现）
module BiTreeLSTM4CateDisam (
    buildSampleSet4CateDisam,
    testPipeline4CateDisam
) where

import qualified Numeric.LinearAlgebra as LA
import qualified Numeric.LinearAlgebra.Devel as LAD
import Numeric.LinearAlgebra (Vector, Matrix, (#>), outer, tr', cmap, konst, size, sumElements, maxElement, maxIndex, vjoin, subVector, scale)

import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import           Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified System.IO.Streams as S
import qualified Data.String as DS
import Data.Map.Strict (Map)
import Data.Map.Strict as Map hiding (map, foldl, size, filter, splitAt, foldl', take, drop, (\\))
import Debug.Trace

import System.Random (randomRIO, mkStdGen, randomR, StdGen, newStdGen)
import List.Shuffle (shuffle_)
import Data.Maybe (fromMaybe)
import Text.Printf (printf)
import Data.List (foldl', foldl, sum, elemIndex, elemIndices, intersect, (\\))
import Data.Tuple.Utils (fst3, snd3, thd3)
import Control.Monad (when, foldM)
import GHC.Generics (Generic)
import Control.Exception (SomeException, bracket, catch)
import System.IO (readFile, writeFile)
import Data.Aeson (encode, decode)
import Database.MySQL.Base

import Utils hiding (isLeafNode)
import Category (Category(..))
import AmbiResol (Prior(..), PhraSyn0, PhraSyn0se, stringToCTPList, priorWithHighestFreq, getPhraSyn0FromStr,readPhraSyn0seFromStr)
import Phrase (Tag, PhraStru, Seman)
import CL
import Statistics (cate2Vec, tag2Vec, stru2Vec)
import Database
import Rule

-- ===================== 访问MySQL表的JSON值 =====================
{- 在aeson中, [Double]是ToJSON、FromJSON的实例.
 - encode [1.0,2.0,3.0]  ≡ "[1.0,2.0,3.0]" :: LazyByteString
 - 辅助：[Double] <-> ByteString <-> Text (用于 MySQLText 传输)
 -}

-- 实现：[Double] --> ByteString --> Text
doubleArrToText :: [Double] -> Text
doubleArrToText = TE.decodeUtf8 . BL.toStrict . encode

-- 实现: Text --> ByteString --> [Double]
textToDoubleArr :: Text -> Maybe [Double]
textToDoubleArr t = decode $ BL.fromStrict $ TE.encodeUtf8 t

-- 写入数据库：[Double] → MySQLValue (MySQLText)
toMySQLJSON :: [Double] -> MySQLValue
toMySQLJSON = MySQLText . doubleArrToText

-- 读取数据库：MySQLValue → Maybe [Double]
fromMySQLJSON :: MySQLValue -> Maybe [Double]
fromMySQLJSON (MySQLText t) = textToDoubleArr t
fromMySQLJSON _ = Nothing

-- ===================== 辅助函数 ====================

-- Different with that in Module Utils, Empty tree is thought as a leaf.
isLeafNode :: BiTree a -> Bool
isLeafNode Empty = True                  -- Easy to processing
isLeafNode (Node _ l r) = isLeafNode l && isLeafNode r

-- ===================== 动作的多热编码 =====================
{- Convert categorial conversions [Rule] to multiple hot code.
 - This is recursive procedure, and original multiple hots should be initialized as zero vector.
 -}
actionToMultiHot :: [Rule] -> Vector Double -> Vector Double
actionToMultiHot [] origMultiHot = origMultiHot
actionToMultiHot (r:rs) origMultiHot =
  case elemIndex (show r) ccTags of
    Nothing -> actionToMultiHot rs origMultiHot
    Just idx ->
      let lst = LA.toList origMultiHot
          lst' = take idx lst ++ 1.0 : drop (idx+1) lst
          vec = LA.fromList lst'
      in actionToMultiHot rs vec

-- Hot is 0.0 or 1.0, respectively corresponds to denying or allowing a categorial conversion.
multiHotToAction :: Vector Double -> [Rule]
multiHotToAction multiHotVec = [getRuleFromStr (ccTags!!x) | x <- indices]
    where
    indices = elemIndices True $ map (\x -> case x of
                                              x | abs (x - 1.0) < 1e-5 -> True
                                              x | abs (x - 0.0) < 1e-5 -> False
                                              _ -> error $ "multiHotToAction: x is not 0.0 or 1.0, x = " ++ show x
                                     ) $ LA.toList multiHotVec

{- Convert prediction to categorial conversions [Rule].
 - Hot threshhold is a value, those bigger than it mean allowing corresponding categorial conversions, and those less than it mean denying.
 - Use 1.0 to represent allowing, and use 0.0 to represent denying.
 -}
vecToAction :: Vector Double -> Double -> [Rule]
vecToAction vec hotThreshhold = multiHotToAction multiHotVec
    where
    multiHotVec = LA.fromList $ map (\x -> case x > hotThreshhold of
                                             True -> 1.0
                                             False -> 0.0
                                    ) $ LA.toList vec

-- Convert prediction to multiple hots code.
vecToMultiHot :: Vector Double -> Double -> Vector Double
vecToMultiHot vec hotThreshhold = LA.fromList $ map (\x -> case x > hotThreshhold of
                                                             True -> 1.0
                                                             False -> 0.0
                                                    ) $ LA.toList vec

-- ===================== 语法-语义二叉树的定义 =====================
type NodeFeat = Vector Double            -- Embedding vector, [Double], of a word or a phrase.
data SyntaxTree a = EmptySTree
                       | STreeNode
                           { nodeData :: a
                           , stLeft   :: SyntaxTree a
                           , stRight  :: SyntaxTree a
                           , stIsLeaf :: Bool
                           } deriving (Show, Eq)
type RawSyntaxTree = SyntaxTree NodeFeat

-- 训练样本：(二叉树, 目标动作)
type TrainSample = (RawSyntaxTree, [Rule])

-- JSON 样本包装结构
data SampleRecord = SampleRecord
                    { recTree  :: RawSyntaxTree
                    , recTargetAct :: [Rule]
                    } deriving (Show, Generic)

-- ===================== 激活函数与导数（修复接口一致性） =====================
-- 激活函数 σ
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

-- 激活函数 σ 的向量版（逐元素计算 σ）
vecSigmoid :: Vector Double -> Vector Double
vecSigmoid = cmap sigmoid

-- 激活函数 σ 的导数的向量版（逐元素计算 σ 导数）
sigmoidDeriv :: Vector Double -> Vector Double
sigmoidDeriv x =
  let v1 = konst 1 (size x) :: Vector Double
      sigVec = cmap sigmoid x
  in sigVec * (v1 - sigVec)

-- 双曲函数 tanh 的向量版
vecTanh :: Vector Double -> Vector Double
vecTanh = cmap tanh

-- 双曲函数 tanh 的导数的向量版
tanhDeriv :: Vector Double -> Vector Double
tanhDeriv z = cmap (\x -> 1 - tanh x * tanh x) z

-- 线性整流函数（修正线性单元）relu 的导数的向量版
vecReLU :: Vector Double -> Vector Double
vecReLU = cmap (\x -> max 0 x)

-- 线性整流函数（修正线性单元）relu 的导数的向量版
reluDeriv :: Vector Double -> Vector Double
reluDeriv = cmap (\x -> if x>0 then 1.0 else 0.0)

-- 软性最大值函数（Softmax），带有减去最大值防止 exp 溢出。
softmax :: Vector Double -> Vector Double
softmax vec =
  let m = LA.maxElement vec
      expVec = cmap (\x -> exp (x - m)) vec
      sumExp = sumElements expVec
      sumExpClip = max sumExp 1e-12
  in cmap (/ sumExpClip) expVec

{- 损失的使用logits的二进制交叉熵（BCEWithLogits）
 - BCEWithLogits(z,y) = max(0,z) - y * z + log(1 + exp(-1.0 * abs(z)))
 -}
binaryCrossEntropyWithLogits :: Vector Double -> Vector Double -> (Vector Double, Double)
binaryCrossEntropyWithLogits logits label
  | LA.size logits /= LA.size label = error "binaryCrossEntropyWithLogits: logits/label dimension mismatch."
  | otherwise = (lossVec, lossTotal)
      where
      max0z = cmap (max 0.0) logits
      negyz = cmap negate $ label * logits
      log1z = cmap (log . ((+) 1.0) . exp . negate . abs) logits
      lossVec = max0z + negyz + log1z
      lossTotal = LA.sumElements lossVec

-- ===================== 网络参数 & 梯度结构 =====================
data NetParam = NetParam
  { leafProjW    :: Matrix Double
  , leafProjB    :: Vector Double
  , nonLeafProjW :: Matrix Double
  , nonLeafProjB :: Vector Double
  , inpW         :: Matrix Double
  , inpLW        :: Matrix Double
  , inpRW        :: Matrix Double
  , inpB         :: Vector Double
  , forgetLW      :: Matrix Double
  , forgetLLW     :: Matrix Double
  , forgetLRW     :: Matrix Double
  , forgetLB      :: Vector Double
  , forgetRW      :: Matrix Double
  , forgetRLW     :: Matrix Double
  , forgetRRW     :: Matrix Double
  , forgetRB      :: Vector Double
  , outW         :: Matrix Double
  , outLW        :: Matrix Double
  , outRW        :: Matrix Double
  , outB         :: Vector Double
  , cellCandW    :: Matrix Double
  , cellCandLW   :: Matrix Double
  , cellCandRW   :: Matrix Double
  , cellCandB    :: Vector Double
  , clsWeight    :: Matrix Double
  , clsBias      :: Vector Double
  } deriving (Show)

-- 求隐层维度
netHidDim :: NetParam -> Int
netHidDim NetParam{inpW = w} = fst (LA.size w)

-- 求投影层维度
netProjDim :: NetParam -> Int
netProjDim netParam = fst . LA.size $ leafProjW netParam

-- 梯度网络，就是参数网络。每个参数都有损失Loss对它的偏导数，dLdp。所有偏导数，组成梯度，形成最快下降方向。
type NetGrad = NetParam

-- 基本记忆单元的状态包括：隐状态 h 和记忆状态 c。整个 LSTM 网络的基本记忆单元有 hidDim 个，hidDim 是隐层维度。
-- 下一个时间步（父节点）的状态依赖当前时间步（子节点）的状态。
data LSTMState = LSTMState
                 { hState :: Vector Double
                 , cState :: Vector Double
                 } deriving (Show)

-- 空的 LSTM 状态：所有记忆单元的 h 和 c 都是 0.0
emptyLSTMState :: Int -> LSTMState
emptyLSTMState d = LSTMState (konst 0.0 d) (konst 0.0 d)

-- 前向计算到一个时间步（节点）时，存储计算结果，后者用于后向梯度计算。
data NodeCache = NodeCache
  { ncFeat      :: NodeFeat          -- 节点的特征向量, {cateVec; wordVec} 或 {cateVec; tagVec; struVec}
  , ncProjFeat  :: Vector Double     -- 投影层输出向量, 维度是 projDim
  , ncInpGatePre :: Vector Double    -- 输入门激活前的值 z 的向量
  , ncForgetLPre :: Vector Double    -- 左遗忘门激活前的值 z 的向量
  , ncForgetRPre :: Vector Double    -- 右遗忘门激活前的值 z 的向量
  , ncOutGatePre :: Vector Double    -- 输出门激活前的值 z 的向量
  , ncCellCandPre :: Vector Double   -- 候选的细胞状态激活前的值 z 的向量
  , ncInpGate   :: Vector Double     -- 输入门（激活后）输出 a
  , ncForgetLG  :: Vector Double     -- 左遗忘门（激活后）输出 a
  , ncForgetRG  :: Vector Double     -- 右遗忘门（激活后）输出 a
  , ncOutGate   :: Vector Double     -- 输出门（激活后）输出 a
  , ncCellCand  :: Vector Double     -- 候选的细胞状态（激活后）的值 a
  , ncNewC      :: Vector Double     -- 更新后的细胞状态
  , ncNewH      :: Vector Double     -- 更新后的隐状态
  , ncLeftH     :: Vector Double     -- 当前节点收到的左孩子隐状态
  , ncLeftC     :: Vector Double     -- 当前节点收到的左孩子细胞状态
  , ncRightH    :: Vector Double     -- 当前节点收到的右孩子隐状态
  , ncRightC    :: Vector Double     -- 当前节点收到的右孩子细胞状态
  , ncIsLeaf    :: Bool              -- 当前节点是否叶子的显性标记
  } deriving (Show)

getRootCache :: SyntaxTree NodeCache -> NodeCache
getRootCache (STreeNode c _ _ _) = c
getRootCache EmptySTree = error "getRootCache: empty tree has no cache"

-- 前向计算结果：根节点的状态（h, c）、与句法树同构的前向计算结果（每个节点有它的NodeCache实例）
data TreeFwdResult = TreeFwdResult
  { tfRootState :: LSTMState
  , tfCacheTree :: SyntaxTree NodeCache
  } deriving (Show)

{- 根据是否叶子，把输入的原始特征投影到给定维度的空间，这是一个线性的仿射变换。
 - 输入的原始特征，对于叶子，是 {cateVec;wordVec}；对于非叶子，是 {cateVec;tagVec;struVec}。
 -}
projectRawFeat :: NetParam -> Bool -> Vector Double -> Vector Double
projectRawFeat NetParam{..} isLeaf rawFeat =
  let
    expectedRawDim = if isLeaf
      then snd $ LA.size leafProjW
      else snd $ LA.size nonLeafProjW
    actualDim = size rawFeat
  in
    if actualDim /= expectedRawDim
      then error $ "projectRawFeat: feature dimension mismatch, expected " ++ show expectedRawDim ++ ", got " ++ show actualDim
      else if isLeaf
        then leafProjW #> rawFeat + leafProjB
        else nonLeafProjW #> rawFeat + nonLeafProjB

-- ===================== 参数初始化 =====================
-- 随机向量
randVec :: Int -> StdGen -> (Vector Double, StdGen)
randVec n gen
  | n < 0    = error "randVec: vector size >= 0 required"
  | otherwise =
      let go :: Int -> StdGen -> [Double] -> ([Double], StdGen)
          go 0 g acc = (reverse acc, g)
          go k g acc =
              let (x, g') = randomR (-0.1, 0.1) g
              in go (k-1) g' (x : acc)
          (xs, gen') = go n gen []
          vec = LA.fromList xs
      in (vec, gen')

-- 随机矩阵
randMat :: Int -> Int -> StdGen -> (Matrix Double, StdGen)
randMat rows cols gen
  | rows < 0 || cols < 0 = error "randMat: rows/cols >=0 required"
  | otherwise =
      let total = rows * cols
          (vec, g') = randVec total gen
          mat = LA.reshape cols vec
      in (mat, g')

-- 初始化网络参数
initNetParam :: Int -> Int -> Int -> Int -> Int -> IO NetParam
initNetParam leafRawDim nonLeafRawDim projDim hidDim labelTotal = do
  let
    g8 = mkStdGen 42                      -- 生成随机函数的种子
    (lProjW, g9) = randMat projDim leafRawDim g8
    (lProjB, g10) = randVec projDim g9
    (nlProjW, g11) = randMat projDim nonLeafRawDim g10
    (nlProjB, g12) = randVec projDim g11

    (iW, g13) = randMat hidDim projDim g12      -- 输入门权重矩阵，用于右乘投影层输出向量
    (iLW, g14) = randMat hidDim hidDim g13      -- 输入门左隐矩阵，用于右乘左孩子隐状态向量
    (iRW, g15) = randMat hidDim hidDim g14      -- 输入门右隐矩阵，用于右乘右孩子隐状态向量
    (iB, g16) = randVec hidDim g15              -- 输入门偏置向量

    (fLW, g17) = randMat hidDim projDim g16     -- 遗忘门权重矩阵，用于右乘投影层输出向量
    (fLLW, g18) = randMat hidDim hidDim g17     -- 遗忘门左隐矩阵，用于右乘左孩子隐状态向量
    (fLRW, g19) = randMat hidDim hidDim g18     -- 遗忘门右隐矩阵，用于右乘右孩子隐状态向量
    (fLB, g20) = randVec hidDim g19             -- 遗忘门偏置向量

    (fRW, g21) = randMat hidDim projDim g20     -- 遗忘门权重矩阵，用于右乘投影层输出向量
    (fRLW, g22) = randMat hidDim hidDim g21     -- 遗忘门左隐矩阵，用于右乘左孩子隐状态向量
    (fRRW, g23) = randMat hidDim hidDim g22     -- 遗忘门右隐矩阵，用于右乘右孩子隐状态向量
    (fRB, g24) = randVec hidDim g23             -- 遗忘门偏置向量

    (oW, g25) = randMat hidDim projDim g24      -- 输出门权重矩阵，用于右乘投影层输出向量
    (oLW, g26) = randMat hidDim hidDim g25      -- 输出门左隐矩阵，用于右乘左孩子隐状态向量
    (oRW, g27) = randMat hidDim hidDim g26      -- 输出门右隐矩阵，用于右乘右孩子隐状态向量
    (oB, g28) = randVec hidDim g27              -- 输出门偏置向量

    (cW, g29) = randMat hidDim projDim g28      -- 候选细胞状态权重矩阵，用于右乘投影层输出向量
    (cLW, g30) = randMat hidDim hidDim g29      -- 候选细胞状态左隐矩阵，用于右乘左孩子隐状态向量
    (cRW, g31) = randMat hidDim hidDim g30      -- 候选细胞状态右隐矩阵，用于右乘右孩子隐状态向量
    (cB, g32) = randVec hidDim g31              -- 候选细胞状态偏置向量

    (clsW, g33) = randMat labelTotal hidDim g32      -- 树的根节点的隐状态到多个（labelTotal)独立二分类的投影矩阵
    (clsB, _) = randVec labelTotal g33               -- 多标签分类投影的偏置向量

  return NetParam
    { leafProjW = lProjW, leafProjB = lProjB              -- Project layer for leaf node
    , nonLeafProjW = nlProjW, nonLeafProjB = nlProjB      -- Project layer for non-leaf node
    , inpW = iW, inpLW = iLW, inpRW = iRW, inpB = iB      -- Input gate
    , forgetLW = fLW, forgetLLW = fLLW, forgetLRW = fLRW, forgetLB = fLB      -- Left forget gate
    , forgetRW = fRW, forgetRLW = fRLW, forgetRRW = fRRW, forgetRB = fRB      -- Right forget gate
    , outW = oW, outLW = oLW, outRW = oRW, outB = oB                          -- Output gate
    , cellCandW = cW, cellCandLW = cLW, cellCandRW = cRW, cellCandB = cB      -- Cell state
    , clsWeight = clsW, clsBias = clsB                                        -- Classification layer
    }

-- 清零网络参数的梯度，即损失 Loss 对每个参数的偏导数都是 0.0
zeroGrad :: Int -> Int -> Int -> Int -> Int -> NetGrad
zeroGrad leafRawDim nonLeafRawDim projDim hidDim labelTotal =
  let zm r c = konst 0.0 (r, c)
      zv d = konst 0.0 d
  in NetParam
    { -- Project layer grad zero
      leafProjW = zm projDim leafRawDim, leafProjB = zv projDim
    , nonLeafProjW = zm projDim nonLeafRawDim, nonLeafProjB = zv projDim
      -- LSTM basic unit grad zero
    , inpW  = zm hidDim projDim, inpLW = zm hidDim hidDim, inpRW = zm hidDim hidDim, inpB = zv hidDim
    , forgetLW = zm hidDim projDim, forgetLLW = zm hidDim hidDim, forgetLRW = zm hidDim hidDim, forgetLB = zv hidDim
    , forgetRW = zm hidDim projDim, forgetRLW = zm hidDim hidDim, forgetRRW = zm hidDim hidDim, forgetRB = zv hidDim
    , outW  = zm hidDim projDim, outLW = zm hidDim hidDim, outRW = zm hidDim hidDim, outB = zv hidDim
    , cellCandW = zm hidDim projDim, cellCandLW = zm hidDim hidDim, cellCandRW = zm hidDim hidDim, cellCandB = zv hidDim
    , clsWeight = zm labelTotal hidDim, clsBias = zv labelTotal
    }

-- ================= 梯度的辅助函数 =================
-- 网络参数的梯度加
gradAdd :: NetGrad -> NetGrad -> NetGrad
gradAdd g1 g2 = NetParam
    { leafProjW = leafProjW g1 + leafProjW g2         -- Projection gradient addition
    , leafProjB = leafProjB g1 + leafProjB g2
    , nonLeafProjW = nonLeafProjW g1 + nonLeafProjW g2
    , nonLeafProjB = nonLeafProjB g1 + nonLeafProjB g2
    , inpW = inpW g1 + inpW g2                        -- LSTM basic unit gradient addition
    , inpLW = inpLW g1 + inpLW g2
    , inpRW = inpRW g1 + inpRW g2
    , inpB = inpB g1 + inpB g2
    , forgetLW = forgetLW g1 + forgetLW g2
    , forgetLLW = forgetLLW g1 + forgetLLW g2
    , forgetLRW = forgetLRW g1 + forgetLRW g2
    , forgetLB = forgetLB g1 + forgetLB g2
    , forgetRW = forgetRW g1 + forgetRW g2
    , forgetRLW = forgetRLW g1 + forgetRLW g2
    , forgetRRW = forgetRRW g1 + forgetRRW g2
    , forgetRB = forgetRB g1 + forgetRB g2
    , outW = outW g1 + outW g2
    , outLW = outLW g1 + outLW g2
    , outRW = outRW g1 + outRW g2
    , outB = outB g1 + outB g2
    , cellCandW = cellCandW g1 + cellCandW g2
    , cellCandLW = cellCandLW g1 + cellCandLW g2
    , cellCandRW = cellCandRW g1 + cellCandRW g2
    , cellCandB = cellCandB g1 + cellCandB g2
    , clsWeight = clsWeight g1 + clsWeight g2
    , clsBias = clsBias g1 + clsBias g2
    }

-- 网络参数的梯度缩放
gradScale :: Double -> NetGrad -> NetGrad
gradScale s g1 =
   NetParam
    { leafProjW = scale s (leafProjW g1)
    , leafProjB = scale s (leafProjB g1)
    , nonLeafProjW = scale s (nonLeafProjW g1)
    , nonLeafProjB = scale s (nonLeafProjB g1)
    , inpW = scale s (inpW g1)
    , inpLW = scale s (inpLW g1)
    , inpRW = scale s (inpRW g1)
    , inpB = scale s (inpB g1)
    , forgetLW = scale s (forgetLW g1)
    , forgetLLW = scale s (forgetLLW g1)
    , forgetLRW = scale s (forgetLRW g1)
    , forgetLB = scale s (forgetLB g1)
    , forgetRW = scale s (forgetRW g1)
    , forgetRLW = scale s (forgetRLW g1)
    , forgetRRW = scale s (forgetRRW g1)
    , forgetRB = scale s (forgetRB g1)
    , outW = scale s (outW g1)
    , outLW = scale s (outLW g1)
    , outRW = scale s (outRW g1)
    , outB = scale s (outB g1)
    , cellCandW = scale s (cellCandW g1)
    , cellCandLW = scale s (cellCandLW g1)
    , cellCandRW = scale s (cellCandRW g1)
    , cellCandB = scale s (cellCandB g1)
    , clsWeight = scale s (clsWeight g1)
    , clsBias   = scale s (clsBias g1)
    }

-- 使梯度 L2 不超过给定的值（maxNorm）
clipGrad :: Double -> NetGrad -> NetGrad
clipGrad maxNorm g =
  let vecNorm v = sqrt $ sumElements (v * v)     -- 欧几里得范数 L2，若元素是某一维度的值间距离，L2 就是欧式距离。
      matNorm m = sqrt $ sumElements (m * m)     -- 求矩阵的 L2
      collectNorms = [ matNorm (leafProjW g), vecNorm (leafProjB g)
                     , matNorm (nonLeafProjW g), vecNorm (nonLeafProjB g)
                     , matNorm (inpW g), matNorm (inpLW g), matNorm (inpRW g), vecNorm (inpB g)
                     , matNorm (forgetLW g), matNorm (forgetLLW g), matNorm (forgetLRW g), vecNorm (forgetLB g)
                     , matNorm (forgetRW g), matNorm (forgetRLW g), matNorm (forgetRRW g), vecNorm (forgetRB g)
                     , matNorm (outW g), matNorm (outLW g), matNorm (outRW g), vecNorm (outB g)
                     , matNorm (cellCandW g), matNorm (cellCandLW g), matNorm (cellCandRW g), vecNorm (cellCandB g)
                     , matNorm (clsWeight g), vecNorm (clsBias g)
                     ]
      totalNorm = sqrt $ sum (map (^2) collectNorms)             -- 网络参数的所有偏导数（即梯度）的 L2
      scaleFactor = if totalNorm > maxNorm then maxNorm / totalNorm else 1.0
  in gradScale scaleFactor g

-- 按学习率、梯度更新网络参数
updateParam :: Double -> NetGrad -> NetParam -> NetParam
updateParam lr grad p =
  let
    updateMat m gm = m + scale (-lr) gm          -- 更新矩阵的方法。若梯度为正，就减小参数；反之，就增大参数。
    updateVec v gv = v + scale (-lr) gv          -- 更新向量
  in p
    { leafProjW    = updateMat (leafProjW p) (leafProjW grad)         -- Update gradient of projection layer
    , leafProjB    = updateVec (leafProjB p) (leafProjB grad)
    , nonLeafProjW = updateMat (nonLeafProjW p) (nonLeafProjW grad)
    , nonLeafProjB = updateVec (nonLeafProjB p) (nonLeafProjB grad)
    , inpW  = updateMat (inpW p) (inpW grad)                          -- Update gradient of LSTM basic unit
    , inpLW  = updateMat (inpLW p) (inpLW grad)
    , inpRW  = updateMat (inpRW p) (inpRW grad)
    , inpB  = updateVec (inpB p) (inpB grad)
    , forgetLW = updateMat (forgetLW p) (forgetLW grad)
    , forgetLLW = updateMat (forgetLLW p) (forgetLLW grad)
    , forgetLRW = updateMat (forgetLRW p) (forgetLRW grad)
    , forgetLB = updateVec (forgetLB p) (forgetLB grad)
    , forgetRW = updateMat (forgetRW p) (forgetRW grad)
    , forgetRLW = updateMat (forgetRLW p) (forgetRLW grad)
    , forgetRRW = updateMat (forgetRRW p) (forgetRRW grad)
    , forgetRB = updateVec (forgetRB p) (forgetRB grad)
    , outW  = updateMat (outW p) (outW grad)
    , outLW  = updateMat (outLW p) (outLW grad)
    , outRW  = updateMat (outRW p) (outRW grad)
    , outB  = updateVec (outB p) (outB grad)
    , cellCandW = updateMat (cellCandW p) (cellCandW grad)
    , cellCandLW = updateMat (cellCandLW p) (cellCandLW grad)
    , cellCandRW = updateMat (cellCandRW p) (cellCandRW grad)
    , cellCandB = updateVec (cellCandB p) (cellCandB grad)
    , clsWeight = updateMat (clsWeight p) (clsWeight grad)
    , clsBias   = updateVec (clsBias p) (clsBias grad)
    }

{- ===================== Binary Tree-LSTM 前向传播 =====================
 - Computing output of current node. Input 'feat' is xw or xp, here xw = {cateVec;wordVec}, and xp = {cateVec;tagVec;struVec}.
 -}
treeForwardCached :: NetParam -> RawSyntaxTree -> TreeFwdResult
treeForwardCached param EmptySTree =         -- 空节点不参与参数更新，缓存仅占位
  let
    hidDim = netHidDim param
  in TreeFwdResult (emptyLSTMState hidDim) EmptySTree
treeForwardCached param st@STreeNode{nodeData = feat, stLeft = l, stRight = r, stIsLeaf = isLeafFlag} =
  let
    leftRes = treeForwardCached param l                -- Firstly, recursively forward computing on left subtree,
    rightRes = treeForwardCached param r               -- and right subtree.
    LSTMState hl cl = tfRootState leftRes              -- Secondly, extract hidden layer vector and status vector.
    LSTMState hr cr = tfRootState rightRes
    NetParam{..} = param
    projFeat = projectRawFeat param isLeafFlag feat

    -- ========== 输入门/输出门/cell候选 并入左右子树隐状态 ==========
    inpGatePre     = inpW #> projFeat + inpLW #> hl + inpRW #> hr + inpB
    forgetLPre     = forgetLW #> projFeat + forgetLLW #> hl + forgetLRW #> hr + forgetLB
    forgetRPre     = forgetRW #> projFeat + forgetRLW #> hl + forgetRRW #> hr + forgetRB
    outGatePre     = outW #> projFeat + outLW #> hl + outRW #> hr + outB
    cellCandPre    = cellCandW #> projFeat + cellCandLW #> hl + cellCandRW #> hr + cellCandB

    inpGate     = vecSigmoid inpGatePre
    forgetLGate = vecSigmoid forgetLPre
    forgetRGate = vecSigmoid forgetRPre
    outGate     = vecSigmoid outGatePre
    cellCand    = vecTanh cellCandPre

    newC = inpGate * cellCand + forgetLGate * cl + forgetRGate * cr
    newH = outGate * vecTanh newC

    cache = NodeCache
      { ncFeat = feat
      , ncProjFeat = projFeat
      , ncInpGatePre = inpGatePre
      , ncForgetLPre = forgetLPre
      , ncForgetRPre = forgetRPre
      , ncOutGatePre = outGatePre
      , ncCellCandPre = cellCandPre
      , ncInpGate = inpGate, ncForgetLG = forgetLGate, ncForgetRG = forgetRGate
      , ncOutGate = outGate, ncCellCand = cellCand, ncNewC = newC, ncNewH = newH
      , ncLeftH = hl, ncLeftC = cl, ncRightH = hr, ncRightC = cr
      , ncIsLeaf = isLeafFlag
      }
    cachedNode = STreeNode cache (tfCacheTree leftRes) (tfCacheTree rightRes) isLeafFlag    -- 函数 tfCacheTree 读取子树前向计算结果的缓存树 STreeNode
  in TreeFwdResult (LSTMState newH newC) cachedNode

-- ===================== 反向传播 Node BP =====================
-- 对一个节点（时间步），已知损失Loss对节点隐状态 h 和对细胞状态 c 的偏导数，用链式规则求 Loss 对各参数的偏导数
backwardNode :: NetParam                       -- 当前的网络参数，其实就是基本记忆单元，它被用于所有样本（双二叉树）的前向计算
             -> NodeCache                      -- 缓存的前向计算结果
             -> Vector Double                  -- Loss 对本节点隐状态的偏导数
             -> Vector Double                  -- Loss 对本节点细胞状态的偏导数
             -> (NetGrad, Vector Double, Vector Double, Vector Double, Vector Double, Vector Double)
                      --  dLdhL          dLdcL          dLdhR          dLdcR          dLdRawFeat
backwardNode param@NetParam{..} nc@NodeCache{..} dLdh dLdc =
  let
    tanhNewC = vecTanh ncNewC                  -- 对细胞状态 c 计算双曲函数 tanh
    dTanhC = tanhDeriv ncNewC                  -- 对细胞状态 c 计算双曲函数的导数 tanh'
    dhdnewC = ncOutGate * dTanhC               -- h = o * tanh(c), dhdnewC = o * tanh'(newC)
    dLdnewC = dLdh * dhdnewC + dLdc            -- 父节点传下来的 dLdc, 必须加上。

    dLdOutGate = dLdh * tanhNewC                            -- dLdo = dLdh * dhdo = dLdh * tanh(c)
    dLdRawOut = dLdOutGate * sigmoidDeriv ncOutGatePre      -- dLdrawO = dLdo * dodrawO = dLdo * sigmoidDeriv ncOutGatePre

    dLdInpGate = dLdnewC * ncCellCand                       -- dLdi = dLdc * dcdi = dLdc * ncCellCand
    dLdRawInp = dLdInpGate * sigmoidDeriv ncInpGatePre      --  dLdrawI = dLdi * didrawI = dLdi * sigmoidDeriv ncInpGatePre

    dLdForgetL = dLdnewC * ncLeftC                          -- dLdfl = dLdc * dcdfl = dLdc * ncLeftC
    dLdRawForgetL = dLdForgetL * sigmoidDeriv ncForgetLPre  -- dLdrawFl = dLdfl * sigmoidDeriv ncForgetLPre

    dLdForgetR = dLdnewC * ncRightC                         -- dLdfr = dLdc * dcdfr = dLdc * ncRightC
    dLdRawForgetR = dLdForgetR * sigmoidDeriv ncForgetRPre  -- dLdrawFr = dLdfr * sigmoidDeriv ncForgetRPre

    dLdCellCand = dLdnewC * ncInpGate                       -- dLdcc = dLdc * ncInpGate
    dLdRawCell = dLdCellCand * tanhDeriv ncCellCandPre      -- dLdrawC = dLdcc * tanhDeriv ncCellCandPre

    dLdProjFeat = tr' inpW #> dLdRawInp                     -- inpW #> projFeat
                + tr' outW #> dLdRawOut                     -- outW #> projFeat
                + tr' cellCandW #> dLdRawCell               -- cellCandW #> projFeat
                + tr' forgetLW #> dLdRawForgetL             -- forgetLW #> projFeat
                + tr' forgetRW #> dLdRawForgetR             -- forgetRW #> projFeat

    dLdhL = tr' inpLW #> dLdRawInp                          -- hl joins calculating preactivations of all gates
          + tr' forgetLLW #> dLdRawForgetL                  -- by left multiplying inpLW, forgetLLW, forgetRLW, outLW, and cellCandLW
          + tr' forgetRLW #> dLdRawForgetR
          + tr' outLW #> dLdRawOut
          + tr' cellCandLW #> dLdRawCell

    dLdcL = dLdnewC * ncForgetLG                            -- cl joins calculating NEW preactivation of cellCand gate.
                                                            -- dLdcL = dLdnewC * dnewCdcL = dLdnewC * ncForgetLG
    dLdhR = tr' inpRW #> dLdRawInp                          -- hr joins calculating preactivations of all gates
          + tr' forgetLRW #> dLdRawForgetL                  -- by left multiplying inpRW, forgetLRW, forgetRRW, outRW, and cellCandRW
          + tr' forgetRRW #> dLdRawForgetR
          + tr' outRW #> dLdRawOut
          + tr' cellCandRW #> dLdRawCell

    dLdcR = dLdnewC * ncForgetRG                            -- dLdcR = dLdnewC * dnewCdcR = dLdnewC * ncForgetRG

    gradInpW = outer dLdRawInp ncProjFeat
    gradInpLW = outer dLdRawInp ncLeftH
    gradInpRW = outer dLdRawInp ncRightH
    gradInpB = dLdRawInp

    gradForgetLW = outer dLdRawForgetL ncProjFeat
    gradForgetLLW = outer dLdRawForgetL ncLeftH
    gradForgetLRW = outer dLdRawForgetL ncRightH
    gradForgetLB = dLdRawForgetL

    gradForgetRW = outer dLdRawForgetR ncProjFeat
    gradForgetRLW = outer dLdRawForgetR ncLeftH
    gradForgetRRW = outer dLdRawForgetR ncRightH
    gradForgetRB = dLdRawForgetR

    gradOutW = outer dLdRawOut ncProjFeat
    gradOutLW = outer dLdRawOut ncLeftH
    gradOutRW = outer dLdRawOut ncRightH
    gradOutB = dLdRawOut

    gradCellCandW = outer dLdRawCell ncProjFeat
    gradCellCandLW = outer dLdRawCell ncLeftH
    gradCellCandRW = outer dLdRawCell ncRightH
    gradCellCandB = dLdRawCell

    leafRawDim = snd $ LA.size leafProjW
    nonLeafRawDim = snd $ LA.size nonLeafProjW
    projDim = LA.size leafProjB
    hidDim = LA.size inpB
    labelTotal = fst $ LA.size clsWeight
    zeroG = zeroGrad leafRawDim nonLeafRawDim projDim hidDim labelTotal         -- Zero gradient network.

    (projGrad, dLdRawFeat)
      | ncIsLeaf =
          let
              gW = outer dLdProjFeat ncFeat            -- ncFeat = {cateVec; wordVec}
              gB = dLdProjFeat
              dX = tr' leafProjW #> dLdProjFeat
              pg = zeroG { leafProjW = gW, leafProjB = gB }    -- Projection layer gradient
          in (pg, dX)
      | otherwise =
          let gW = outer dLdProjFeat ncFeat            -- ncFeat = {cateVec; tagVec; struVec}
              gB = dLdProjFeat
              dX = tr' nonLeafProjW #> dLdProjFeat
              pg = zeroG { nonLeafProjW = gW, nonLeafProjB = gB }
          in (pg, dX)

    lstmGrad = zeroG
      { inpW = gradInpW, inpLW = gradInpLW, inpRW = gradInpRW, inpB = gradInpB
      , forgetLW = gradForgetLW, forgetLLW = gradForgetLLW, forgetLRW = gradForgetLRW, forgetLB = gradForgetLB
      , forgetRW = gradForgetRW, forgetRLW = gradForgetRLW, forgetRRW = gradForgetRRW, forgetRB = gradForgetRB
      , outW = gradOutW, outLW = gradOutLW, outRW = gradOutRW, outB = gradOutB
      , cellCandW = gradCellCandW, cellCandLW = gradCellCandLW, cellCandRW = gradCellCandRW, cellCandB = gradCellCandB
      }

    fullGrad = projGrad `gradAdd` lstmGrad
  in (fullGrad, dLdhL, dLdcL, dLdhR, dLdcR, dLdRawFeat)

-- ===================== 整树反向传播【修复空树梯度】 =====================
-- 对一棵句法树调用本函数，输入dLdh和dLdc，递归执行，直到求出整棵树的梯度和。
backwardTree :: NetParam -> SyntaxTree NodeCache -> Vector Double -> Vector Double -> NetGrad
backwardTree param EmptySTree _ _ =
  let
    leafRawDim = snd $ LA.size (leafProjW param)
    nonLeafRawDim = snd $ LA.size (nonLeafProjW param)
    projDim = fst $ LA.size (leafProjW param)
    hidDim = fst $ LA.size (inpW param)
    labelTotal = fst $ LA.size (clsWeight param)
  in zeroGrad leafRawDim nonLeafRawDim projDim hidDim labelTotal
backwardTree param (STreeNode cache leftTree rightTree _) dLdh dLdc =
  let
    (nodeGrad, dLdhL, dLdcL, dLdhR, dLdcR, _) = backwardNode param cache dLdh dLdc
    gradLeft = backwardTree param leftTree dLdhL dLdcL
    gradRight = backwardTree param rightTree dLdhR dLdcR
    totalGrad = nodeGrad `gradAdd` gradLeft `gradAdd` gradRight
  in totalGrad

-- ===================== 单样本前向+损失+梯度 =====================
sampleForwardLossGrad :: NetParam -> TrainSample -> (Double, NetGrad)
sampleForwardLossGrad param (tree, action) =
  let
    fwd = treeForwardCached param tree
    LSTMState hRoot _ = tfRootState fwd

    NetParam{..} = param
    logits = clsWeight #> hRoot + clsBias
    zeroMultiHot = konst 0.0 (fst $ LA.size clsWeight)
    label = actionToMultiHot action zeroMultiHot
    (_, lossTotal) = binaryCrossEntropyWithLogits logits label
    dLdLogit = vecSigmoid logits - label                   -- dLdz
    gradClsW = outer dLdLogit hRoot
    gradClsB = dLdLogit
    dLdHRoot = tr' clsWeight #> dLdLogit
    grad = backwardTree param (tfCacheTree fwd) dLdHRoot (konst 0.0 (LA.size dLdHRoot))
    totalGrad = grad
      { clsWeight = gradClsW
      , clsBias = gradClsB
      }
  in (lossTotal, totalGrad)

-- ===================== 单样本前向+损失+梯度 =====================
sampleForwardLoss :: NetParam -> TrainSample -> Double
sampleForwardLoss param (tree, action) = lossTotal
  where
    fwd = treeForwardCached param tree
    LSTMState hRoot _ = tfRootState fwd

    NetParam{..} = param
    logits = clsWeight #> hRoot + clsBias
    zeroMultiHot = konst 0.0 (fst $ LA.size clsWeight)
    label = actionToMultiHot action zeroMultiHot
    (_, lossTotal) = binaryCrossEntropyWithLogits logits label

batchTrainStep :: NetParam -> [TrainSample] -> Double -> (Double, NetParam)
batchTrainStep param samples lr =
  let
    leafRawDim = snd $ LA.size $ leafProjW param
    nonLeafRawDim = snd $ LA.size $ nonLeafProjW param
    projDim = netProjDim param
    hidDim = netHidDim param
    labelTotal = fst $ LA.size $ clsWeight param
    initG = zeroGrad leafRawDim nonLeafRawDim projDim hidDim labelTotal
    pairs = map (sampleForwardLossGrad param) samples
    losses = map fst pairs
    grads = map snd pairs
    totalLoss = sum losses
    batchSize = fromIntegral (length samples)
    avgLoss = totalLoss / batchSize
    sumGrad = foldl gradAdd initG grads
    avgGrad = gradScale (1.0 / batchSize) sumGrad
    clippedGrad = clipGrad 5.0 avgGrad
    newParam = updateParam lr clippedGrad param
  in (avgLoss, newParam)

-- ===================== 树构造辅助函数（测试用） =====================
mkLeaf :: [Double] -> [Double] -> RawSyntaxTree
mkLeaf cateVec wordEmb =
  let feat = LA.fromList (cateVec ++ wordEmb)
  in STreeNode feat EmptySTree EmptySTree True

mkNonLeaf :: [Double] -> [Double] -> [Double] -> RawSyntaxTree -> RawSyntaxTree -> RawSyntaxTree
mkNonLeaf _ _ _ EmptySTree EmptySTree = error "mkNonLeaf: Parameter error."
mkNonLeaf cateVec tagVec struVec l r =
  let feat = LA.fromList (cateVec ++ tagVec ++ struVec)
  in STreeNode feat l r False

-- ===================== 训练循环：增加样本 shuffle =====================
trainLoop :: NetParam -> [TrainSample] -> Double -> Int -> IO NetParam
trainLoop param _ _ 0 = return param
trainLoop param samples lr epoch = do
  gen <- mkStdGen <$> randomRIO (1,99999)
  let shuffled = shuffle_ samples gen
      (avgLoss, newParam) = batchTrainStep param shuffled lr
  when (epoch `mod` 50 == 0) $
    putStrLn $ "Epoch剩余:" ++ show epoch ++ " | AvgLoss:" ++ show avgLoss
  trainLoop newParam samples lr (epoch - 1)

-- ===================== 业务数据库接口 =====================
--- | 生成N维向量，固定种子，纯函数，无IO
genVector :: Int -> Int -> [Double]
genVector seed dim = go dim (mkStdGen seed)
  where
    go 0 _ = []
    go n g =
      let (x, g') = randomR (-0.25, 0.25) g
      in x : go (n-1) g'

-- 300维OOV向量，固定seed=42，复现可控
oov300 :: [Double]
oov300 = genVector 42 300

-- 把OOV向量插入数据库表sgns_renmin_word.
createRandomVecForOOV :: IO ()
createRandomVecForOOV = do
    putStrLn $ "维度：" ++ show (length oov300)
    let jsonBs = BL.toStrict $ encode oov300         -- encode :: [Double] -> BL.ByteString
    putStrLn (show jsonBs)
    conn <- getConn
    let sqlstat = DS.fromString $ "INSERT INTO sgns_renmin_word SET word = ?, emb = CONVERT( ? USING utf8mb4 )"
    stmt <- prepareStmt conn sqlstat
    ok <- executeStmt conn stmt [toMySQLText "<OOV>", toMySQLBytes jsonBs]
    putStrLn $ "createRandomVecForOOV: Insert record whose id is " ++ (show . getOkLastInsertID) ok
    closeStmt conn stmt
    close conn

{- ========= 转换 BiTree (PhraSyn0, Seman) 到 SyntaxTree NodeFeat =========
 - data BiTree a = Empty | Node a (BiTree a) (BiTree a) deriving (Eq)
 - data SyntaxTree a = EmptySTree | STreeNode
                                  { nodeData :: a
                                  , stLeft :: SyntaxTree a
                                  , stRight :: SyntaxTree a
                                  , stIsLeaf :: Bool
                                  } deriving (Show, Eq)
 - type PhraSyn0se = (Category, Tag, PhraStru, Seman)
 - Count OOV words and non-OOV words. When embedding vectors of words are not used, oovCnt and noovCnt denote number of leaves and that of non-leaves.
 -}
biTreePhrasyn0se2SyntaxTreeNodeFeat :: BiTree PhraSyn0se
                                     -> Map Category (Vector Double)
                                     -> Map Tag (Vector Double)
                                     -> Map PhraStru (Vector Double)
                                     -> Int
                                     -> Int
                                     -> Int
                                     -> [Double]
                                     -> Int                           -- Num. of OOV words in this tree
                                     -> Int                           -- Num. of non-OOV words in this tree
                                     -> Bool                          -- Use embedding vector of words or not
                                     -> IO (SyntaxTree NodeFeat, Int, Int)
biTreePhrasyn0se2SyntaxTreeNodeFeat Empty _ _ _ _ _ _ _ _ _ _ = return (EmptySTree, 0, 0)
biTreePhrasyn0se2SyntaxTreeNodeFeat (Node phraSyn0se lt rt) cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov oovCnt noovCnt useEmbVec = do
--    putStrLn $ "PhraSyn0se: (" ++ show (fst4 phraSyn0se) ++ ", " ++ snd4 phraSyn0se ++ ", " ++ thd4 phraSyn0se ++ ", " ++ fth4 phraSyn0se ++ ")"
    let
        (v1, g1) = randVec dim_cate (mkStdGen 42)
        (v2, g2) = randVec dim_tag g1
        (v3, _) = randVec dim_stru g2
        cate = fst4 phraSyn0se
        tag = snd4 phraSyn0se                 -- Use 'throwFstLast' on String if there are double quotes bracketing it.
        stru = thd4 phraSyn0se                -- Use 'throwFstLast' on String if there are double quotes bracketing it.
        seman = fth4 phraSyn0se               -- Use 'throwFstLast' on String if there are double quotes bracketing it.
        semanTerm = getTermFromStr seman                       -- CL term of logic semantics
        cateVec' = Map.lookup cate cateVecMap                  -- Maybe Double
        tagVec' = Map.lookup tag tagVecMap
        struVec' = Map.lookup stru struVecMap
        cateVec = fromMaybe v1 cateVec'
        tagVec = fromMaybe v2 tagVec'
        struVec = fromMaybe v3 struVec'

    if useEmbVec                 -- Using embedding vector when node is a leaf.
      then
        if isPrimTerm semanTerm
          then do                -- Word seman
--            putStrLn $ "  It's a primitive term."
            let seman = init $ show semanTerm
            confInfo <- readFile "Configuration"
            let wordEmbTbl = getConfProperty "wordEmbTbl" confInfo
            conn <- getConn
            let sqlstat = Query $ DS.fromString ("select CAST(emb AS CHAR) from " ++ wordEmbTbl ++ " where word = ?")
            (defs, is) <- query conn sqlstat [MySQLText (T.pack seman)]       -- Search database table 'wordEmbTbl'
            rows <- S.toList is
--            putStrLn $ "  Size of rows: " ++ show (length rows)
            close conn
            if rows == []                      -- OOV
              then
                return ( STreeNode { nodeData = LA.vjoin [cateVec, LA.fromList oov]
                                   , stLeft = EmptySTree
                                   , stRight = EmptySTree
                                   , stIsLeaf = True
                                   }
                       , oovCnt + 1, noovCnt
                       )
              else do                         -- Not OOV
                let embVec = (fromMaybe' . fromMySQLJSON) ((rows!!0)!!0)     -- [Double]
--                putStrLn $ "  embVec(" ++ seman ++ "): " ++ show embVec
                    nodeFeat = LA.vjoin [cateVec, LA.fromList embVec]
                return ( STreeNode { nodeData = nodeFeat
                                   , stLeft = EmptySTree
                                   , stRight = EmptySTree
                                   , stIsLeaf = True
                                   }
                       , oovCnt, noovCnt + 1
                       )
          else do           -- Phrasal seman
--          putStrLn $ "  It's a compound term."
            (lt, oovCnt', noovCnt') <- biTreePhrasyn0se2SyntaxTreeNodeFeat lt cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov oovCnt noovCnt useEmbVec
            (rt, oovCnt'', noovCnt'') <- biTreePhrasyn0se2SyntaxTreeNodeFeat rt cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov oovCnt' noovCnt' useEmbVec
            return ( STreeNode { nodeData = LA.vjoin [cateVec, tagVec, struVec]
                               , stLeft = lt
                               , stRight = rt
                               , stIsLeaf = False
                               }
                   , oovCnt'', noovCnt''
                   )

      else
        if isPrimTerm semanTerm
          then do
--            putStrLn $ "  Leaf node: " ++ show semanTerm
--            putStrLn $ "  Verify cateVec, tagVec, and struVec: " ++ show (cateVec /= v1 && tagVec == v2 && struVec == v3)
            return ( STreeNode { nodeData = cateVec
                               , stLeft = EmptySTree
                               , stRight = EmptySTree
                               , stIsLeaf = True
                               }
                   , oovCnt + 1, noovCnt
                   )
          else do
--            putStrLn $ " Non-leaf node: " ++ show semanTerm
--            putStrLn $ " Verify cateVec, tagVec, and struVec: " ++ show (cateVec /= v1 && tagVec /= v2 && struVec /= v3)
            (lt, oovCnt', noovCnt') <- biTreePhrasyn0se2SyntaxTreeNodeFeat lt cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov oovCnt noovCnt useEmbVec
            (rt, oovCnt'', noovCnt'') <- biTreePhrasyn0se2SyntaxTreeNodeFeat rt cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov oovCnt' noovCnt' useEmbVec
            return ( STreeNode { nodeData = LA.vjoin [cateVec, tagVec, struVec]
                               , stLeft = lt
                               , stRight = rt
                               , stIsLeaf = False
                               }
                   , oovCnt'', noovCnt'' + 1
                   )

buildSampleSet4CateDisam :: Int -> Int -> Int -> IO [SampleRecord]
buildSampleSet4CateDisam dim_cate dim_tag dim_stru = do
  confInfo <- readFile "Configuration"
  let cate_ambig_resol_source = getConfProperty "cate_ambig_resol_source" confInfo
      cate_resol_sample_startId = getConfProperty "cate_resol_sample_startId" confInfo
      cate_resol_sample_endId = getConfProperty "cate_resol_sample_endId" confInfo
      startId = read cate_resol_sample_startId :: Int
      endId = read cate_resol_sample_endId :: Int
      leafUsingEmbVec = getConfProperty "leafUsingEmbVec" confInfo
      useEmbVec = read leafUsingEmbVec :: Bool

  putStrLn $ " useEmbVec: " ++ leafUsingEmbVec

  conn <- getConn
  let sqlstat = DS.fromString $ "select bitree_phrasyn0se_rules from " ++ cate_ambig_resol_source ++ " where id >= ? && id <= ?"
  stmt <- prepareStmt conn sqlstat
  (defs, is) <- queryStmt conn stmt [toMySQLInt32U startId, toMySQLInt32U endId]
  textList <- readStreamByText [] is        -- [String]
  closeStmt conn stmt
  close conn

  let treeRulesList = map (\x -> ( stringToBiTree readPhraSyn0seFromStr (fst x)           -- BiTree PhraSyn0se
                                 , getRulesFromStr (snd x)                       -- [Rule]
                                 )
                          ) $ map stringToTuple textList

--  putStrLn $ "Sample number = " ++ show (length treeRulesList) ++ ", the first is: " ++ show (treeRulesList!!0)

  cateVecMap <- cate2Vec                         -- Map Category (Vector Double)
  tagVecMap <- tag2Vec                           -- Map Tag (Vector Double)
  struVecMap <- stru2Vec                         -- Map PhraStru (Vector Double)

  sampleSetWithOOVCnt <- mapWithProgressStep 100
      (\x -> do
          (tree, oovCnt, noovCnt) <- biTreePhrasyn0se2SyntaxTreeNodeFeat (fst x) cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov300 0 0 useEmbVec
          let rules = snd x
          return ( SampleRecord{ recTree = tree
                               , recTargetAct = rules
                               }
                  , oovCnt                  -- Total of OOV words in the tree
                  , noovCnt                 -- Total of non-OOV words in the tree
                  )
      ) treeRulesList
  let sampleSet = map fst3 sampleSetWithOOVCnt
      oovTotal = foldl (+) 0 $ map snd3 sampleSetWithOOVCnt
      noovTotal = foldl (+) 0 $ map thd3 sampleSetWithOOVCnt
      prop = 100 * (fromIntegral oovTotal / fromIntegral noovTotal) :: Double
--  putStrLn $ "buildSampleSet4CateDisam: sampleSet!!0: " ++ show (sampleSet!!0)
  putStrLn $ "buildSampleSet4CateDisam: oovTotal = " ++ show oovTotal ++ ", noovTotal = " ++ show noovTotal ++ ", proportion of oov words = " ++ (printf "%.2f" prop) ++ "%"
  return sampleSet

-- 评估整个数据集上模型预测的平均损失率
evalLoss :: NetParam -> [TrainSample] -> Double
evalLoss param samples = avgLoss
  where
    pairs = map (sampleForwardLossGrad param) samples
    losses = map fst pairs
    totalLoss = sum losses
    sampleSetSize = fromIntegral (length samples)
    avgLoss = totalLoss / sampleSetSize

-- ========== 【trainLoop2：支持验证 + 早停 + 保存最优参数】 ==========
-- 入参：初始参数、训练集、验证集、学习率、最大轮数、早停耐心值、批大小
-- 返回：验证集上最优模型参数
trainLoop2 :: NetParam -> [TrainSample] -> [TrainSample] -> Double -> Int -> Int -> Int -> IO NetParam
trainLoop2 initParam trainSet valSet lr maxEpoch patience batchSize = go initParam maxEpoch patience 1 1e9 batchSize
  where
    go :: NetParam -> Int -> Int -> Int -> Double -> Int -> IO NetParam
    go currParam remainEpoch remainPatience epoch currValLoss batchSize
      | remainEpoch <= 0 = do
          putStrLn $ "达到最大 Epoch(" ++ show maxEpoch ++ ")，停止训练，返回最优模型"
          return currParam
      | remainPatience <= 0 = do
          putStrLn $ "早停触发！连续 " ++ show patience ++ " 轮验证损失无下降，终止训练"
          return currParam
      | otherwise = do
          -- 1. 完整一轮训练：遍历训练集更新参数
          newParam <- trainOneEpoch currParam trainSet lr batchSize
          -- 2. 在验证集评估
          let newValLoss = evalLoss newParam valSet
              newTrainLoss = evalLoss newParam trainSet
          putStr $ "Epoch " ++ show epoch ++ " | 训练集损失：" ++ show newTrainLoss ++ ", 验证集损失: " ++ show newValLoss

          confInfo <- readFile "Configuration"
          let cate_ambig_resol_model = getConfProperty "cate_ambig_resol_model" confInfo
              cate_ambig_resol_source = getConfProperty "cate_ambig_resol_source" confInfo
              cate_resol_sample_startId = getConfProperty "cate_resol_sample_startId" confInfo
              cate_resol_sample_endId = getConfProperty "cate_resol_sample_endId" confInfo
              startId = read cate_resol_sample_startId :: Int
              endId = read cate_resol_sample_endId :: Int
              sampleNum = endId - startId + 1
              dim_hidden = getConfProperty "dim_hidden" confInfo
              hot_threshhold = getConfProperty "hot_threshhold" confInfo
              ht = read hot_threshhold :: Double

              lr_str = show $ floor (lr * 1000)
              ht_str = show $ floor (ht * 10)
              tblNameSuffix = "_" ++ cate_ambig_resol_model ++ "_ht" ++ ht_str ++ "_dim" ++ dim_hidden ++ "_lr" ++ lr_str ++ "_ns" ++ show sampleNum
              tblName = cate_ambig_resol_source ++ tblNameSuffix
              sqlstat = DS.fromString $ "INSERT INTO " ++ tblName ++ " set epoch = ?, train_set_loss = ?, val_set_loss = ?"
          conn <- getConn
          stmt <- prepareStmt conn sqlstat
          executeStmt conn stmt [toMySQLInt32U epoch, toMySQLDouble newTrainLoss, toMySQLDouble newValLoss]
          closeStmt conn stmt
          close conn

          -- 3. 判断是否更新最优模型
          if newValLoss < currValLoss
            then do
              putStrLn $ " | 验证损失下降，更新最优模型。"
              go newParam (remainEpoch - 1) patience (epoch + 1) newValLoss batchSize
            else do
              putStrLn $ " | 验证损失未下降，剩余耐心: " ++ show (remainPatience - 1)
              go currParam (remainEpoch - 1) (remainPatience - 1) (epoch + 1) currValLoss batchSize

-- 单轮训练：遍历trainSet，反向传播更新权重
trainOneEpoch :: NetParam -> [TrainSample] -> Double -> Int -> IO NetParam
trainOneEpoch param samples lr batchSize = do
  gen <- mkStdGen <$> randomRIO (1,99999)
  let shuffled = shuffle_ samples gen                                           -- [TrainSample]
      batches = chunk batchSize shuffled                                        -- [[TrainSample]]
      (_, newParam) =
        foldl' (\(_,p) b -> let (loss,p') = batchTrainStep p b lr in (loss,p')) (0.0, param) batches
                                                                                -- Initial loss is set 0.0
  return newParam

--- ===================== 训练与测试：可独立运行 =====================
testPipeline4CateDisam :: IO ()
testPipeline4CateDisam = do
  putStrLn "==== Binary Tree-LSTM ===="
  confInfo <- readFile "Configuration"
  let
    cate_ambig_resol_model = getConfProperty "cate_ambig_resol_model" confInfo
    cate_ambig_resol_source = getConfProperty "cate_ambig_resol_source" confInfo
    cate_resol_sample_startId = getConfProperty "cate_resol_sample_startId" confInfo
    cate_resol_sample_endId = getConfProperty "cate_resol_sample_endId" confInfo
    startId = read cate_resol_sample_startId :: Int
    endId = read cate_resol_sample_endId :: Int

    dim_word_str = getConfProperty "dim_word" confInfo
    dim_cate_str = getConfProperty "dim_cate" confInfo
    dim_tag_str = getConfProperty "dim_tag" confInfo
    dim_stru_str = getConfProperty "dim_stru" confInfo
    dim_proj_str = getConfProperty "dim_proj" confInfo
    dim_hidden_str = getConfProperty "dim_hidden" confInfo
    rule_num_str = getConfProperty "rule_num" confInfo
    hot_threshhold_str = getConfProperty "hot_threshhold" confInfo
    lr_str = getConfProperty "lr" confInfo
    epochs_str = getConfProperty "epochs" confInfo
    earlyStopPatience_str = getConfProperty "earlyStopPatience" confInfo
    batchSize_str = getConfProperty "batchSize" confInfo
    useEmbVec_str = getConfProperty "leafUsingEmbVec" confInfo

    dim_word = read dim_word_str :: Int
    dim_cate = read dim_cate_str :: Int
    dim_tag = read dim_tag_str :: Int
    dim_stru = read dim_stru_str :: Int
    leafRawDim = dim_cate + dim_word
    nonLeafRawDim = dim_cate + dim_tag + dim_stru
    projDim = read dim_proj_str :: Int
    hidDim = read dim_hidden_str :: Int
    hotThreshhold = read hot_threshhold_str :: Double
    ht_str = show $ floor (hotThreshhold * 10)
    labelTotal = read rule_num_str :: Int
    lr = read lr_str :: Double
    epochs = read epochs_str :: Int
    earlyStopPatience = read earlyStopPatience_str :: Int
    batchSize = read batchSize_str :: Int
    useEmbVec = read useEmbVec_str :: Bool

  putStrLn $ " cate_ambig_resol_model: " ++ cate_ambig_resol_model
  putStrLn $ " cate_ambig_resol_source: " ++ cate_ambig_resol_source
  putStrLn $ " cate_resol_sample_startId: " ++ cate_resol_sample_startId
  putStrLn $ " cate_resol_sample_endId: " ++ cate_resol_sample_endId
  putStrLn $ " dim_word: " ++ dim_word_str
  putStrLn $ " dim_cate: " ++ dim_cate_str
  putStrLn $ " dim_tag: " ++ dim_tag_str
  putStrLn $ " dim_stru: " ++ dim_stru_str
  putStrLn $ " leafRawDim: " ++ show leafRawDim
  putStrLn $ " nonLeafRawDim: " ++ show nonLeafRawDim
  putStrLn $ " projDim: " ++ dim_proj_str
  putStrLn $ " hidDim: " ++ dim_hidden_str
  putStrLn $ " hotThreshhold: " ++ hot_threshhold_str
  putStrLn $ " labelTotal: " ++ show labelTotal
  putStrLn $ " lr: " ++ lr_str
  putStrLn $ " epochs: " ++ epochs_str
  putStrLn $ " earlyStopPatience: " ++ earlyStopPatience_str
  putStrLn $ " batchSize: " ++ batchSize_str
  putStrLn $ " useEmbVec: " ++ useEmbVec_str

  let prompt = " Train Binary tree LSTM for category disambiguation in Module BiTreeLSTM4CateDisam, are you sure? [y/n] (RETURN for 'y'): "
  answer <- getLineUntil prompt ["y","n"] True
  if answer == "y"
    then do
      -- ========== 清空存储训练过程损失率的 MySQL 表 ==========
      let
          lr_str = show $ floor (lr * 1000)
          sampleNum = endId - startId + 1
          tblNameSuffix = "_" ++ cate_ambig_resol_model ++ "_ht" ++ ht_str ++ "_dim" ++ dim_hidden_str ++ "_lr" ++ lr_str ++ "_ns" ++ show sampleNum
          tblName = cate_ambig_resol_source ++ tblNameSuffix
          sqlstat1 = DS.fromString $ "SELECT EXISTS(SELECT 1 FROM information_schema.TABLES WHERE TABLE_SCHEMA='ccg4c' AND TABLE_NAME='"
                                     ++ tblName ++ "') AS table_exist"
      conn <- getConn
      stmt <- prepareStmt conn sqlstat1
      (defs, is) <- queryStmt conn stmt []
      rows <- readStreamByInt32U [] is
      case length rows of
        0 -> do
               closeStmt conn stmt
               close conn
               error $ "MySQL table " ++ tblName ++ " does NOT exist."
        x | x > 1 -> do
               closeStmt conn stmt
               close conn
               error $ "Impossible"
        1 -> do
               let prompt = " Records in Table " ++ tblName ++ " will be removed, are you sure? [y/n] (RETURN for 'n'): "
               answer <- getLineUntil prompt ["y","n"] False
               if answer == "y"
                 then do
                   let sqlstat2 = DS.fromString $ "DELETE FROM " ++ tblName
                   stmt <- prepareStmt conn sqlstat2
                   executeStmt conn stmt []
                   closeStmt conn stmt
                   close conn

                   initP <- initNetParam leafRawDim nonLeafRawDim projDim hidDim labelTotal

                   samples0 <- buildSampleSet4CateDisam dim_cate dim_tag dim_stru
                   let samples = map (\x -> (recTree x, recTargetAct x)) samples0
                   putStrLn $ "总样本数量：" ++ show (length samples)

                   -- ========== 8:1:1 数据集划分 ==========
                   gen <- newStdGen
                   let shuffledSamples = shuffle_ samples gen
                       total = length shuffledSamples
                       splitTrain  = floor $ 0.8 * fromIntegral total
                       splitVal    = floor $ 0.9 * fromIntegral total
                       (trainSet, rest) = splitAt splitTrain shuffledSamples
                       (valSet, testSet) = splitAt (splitVal - splitTrain) rest
                   putStrLn $ "训练集(80%)样本数：" ++ show (length trainSet)
                   putStrLn $ "验证集(10%)样本数：" ++ show (length valSet)
                   putStrLn $ "测试集(10%)样本数：" ++ show (length testSet)

                   -- ========== 带验证+早停训练，返回验证最优模型 ==========
                   bestTrainedP <- trainLoop2 initP trainSet valSet lr epochs earlyStopPatience batchSize

                   -- ========== 【最终评估：独立测试集，仅执行一次】 ==========
                   putStrLn "\n====【最终测试集评估】===="
                   case testSet of
                     [] -> putStrLn "警告：测试集为空，无法评估！"
                     _ -> do
                       let totalLoss = evalLoss bestTrainedP testSet
                           zeroMultiHot = konst 0.0 (fst $ LA.size (clsWeight bestTrainedP))
                           predPairs = map (\(tree, targetAct) ->
                                              let
                                                forwResult = treeForwardCached bestTrainedP tree
                                                LSTMState hRoot _ = tfRootState forwResult
                                                logits = clsWeight bestTrainedP #> hRoot + clsBias bestTrainedP
                                                probVec = vecSigmoid logits     -- [Double]
                                                pAct = vecToMultiHot probVec hotThreshhold       -- Vector Double
                                                tAct = actionToMultiHot targetAct zeroMultiHot   -- Vector Double
                                              in (pAct, tAct)
                                           ) testSet                            -- [(Vector Double, Vector Double)]

                           predPairsAct = map (\(pAct, tAct) -> (multiHotToAction pAct, multiHotToAction tAct)) predPairs    -- [([Rule], [Rule])]

                           pActMat = LA.fromRows $ map fst predPairs            -- Matrix Double (length testSet >< labelTotal)
                           tActMat = LA.fromRows $ map snd predPairs            -- Matrix Double (length testSet >< labelTotal)

                        -- True/False: Predict correctly, Positive/Negative: Prediction result
                        -- TP: True Positive, FN: False Negative, FP: False Positive.

                           labelIndices = [0 .. labelTotal - 1]
                           tpNumPerLabel = map (\labelIdx -> length
                                                             $ filter (\(p,t) -> abs (p - 1.0) < 1e-5 && abs (t - 1.0) < 1e-5)
                                                               $ zip (LA.toList ((LA.toColumns pActMat)!!labelIdx)) (LA.toList ((LA.toColumns tActMat)!!labelIdx))
                                               ) labelIndices                   -- Total of true positive instances for each Rule label

                           fnNumPerLabel = map (\labelIdx -> length
                                                             $ filter (\(p,t) -> abs (p - 0.0) < 1e-5 && abs (t - 1.0) < 1e-5)
                                                               $ zip (LA.toList ((LA.toColumns pActMat)!!labelIdx)) (LA.toList ((LA.toColumns tActMat)!!labelIdx))
                                               ) labelIndices                   -- Total of false negative instances for each Rule label

                           fpNumPerLabel = map (\labelIdx -> length
                                                             $ filter (\(p,t) -> abs (p - 1.0) < 1e-5 && abs (t - 0.0) < 1e-5)
                                                               $ zip (LA.toList ((LA.toColumns pActMat)!!labelIdx)) (LA.toList ((LA.toColumns tActMat)!!labelIdx))
                                               ) labelIndices                   -- Total of false positive instances for each Rule label

                           precisionPerLabel = map (\labelIdx -> let
                                                                   tpNum = tpNumPerLabel!!labelIdx
                                                                   fpNum = fpNumPerLabel!!labelIdx
                                                                 in case tpNum + fpNum /= 0 of
                                                                      True -> fromIntegral tpNum / (fromIntegral (tpNum + fpNum) :: Double)
                                                                      False -> 1.0
                                                   ) labelIndices               -- [Double], precision for each Rule label

                           recallPerLabel = map (\labelIdx -> let
                                                                tpNum = tpNumPerLabel!!labelIdx
                                                                fnNum = fnNumPerLabel!!labelIdx
                                                              in case tpNum + fnNum > 0 of
                                                                   True -> fromIntegral tpNum / (fromIntegral (tpNum + fnNum) :: Double)
                                                                   False -> 0.0
                                                ) labelIndices                  -- [Double], recall for each Rule label

                           f1ScorePerLabel = map (\labelIdx -> let
                                                                 precision = precisionPerLabel!!labelIdx
                                                                 recall = recallPerLabel!!labelIdx
                                                               in case precision + recall > 0.0 of
                                                                    True -> 2 * precision * recall / (precision + recall)
                                                                    False -> 0.0
                                                 ) labelIndices                 -- F1Score for each Rule label

                           macro_avg_precision = sum precisionPerLabel / (fromIntegral labelTotal :: Double)
                           macro_avg_recall = sum recallPerLabel / (fromIntegral labelTotal :: Double)
                           macro_avg_f1Score = sum f1ScorePerLabel / (fromIntegral labelTotal :: Double)

                           tpTotal = foldl (+) 0 tpNumPerLabel                  -- [Int]
                           fpTotal = foldl (+) 0 fpNumPerLabel                  -- [Int]
                           fnTotal = foldl (+) 0 fnNumPerLabel                  -- [Int]

                           micro_avg_precision = case tpTotal + fpTotal > 0 of
                             True -> (fromIntegral tpTotal :: Double) / (fromIntegral (tpTotal + fpTotal)) :: Double
                             False -> 1.0
                           micro_avg_recall = (fromIntegral tpTotal :: Double) / (fromIntegral (tpTotal + fnTotal)) :: Double
                           micro_avg_f1Score = 2.0 * micro_avg_precision * micro_avg_recall / (micro_avg_precision + micro_avg_recall)

                           pn_MultiLabel = map (\labelIdx -> let
                                                               tpNum = tpNumPerLabel!!labelIdx
                                                               fnNum = fnNumPerLabel!!labelIdx
                                                             in tpNum + fnNum
                                               ) labelIndices                   -- [Int], Occurence total of positive instances for each Rule label
                           sampleTotal = length testSet
                           weighted_avg_precision = foldl (+) 0.0 (map (\(pn, prec) -> fromIntegral pn * prec) (zip pn_MultiLabel precisionPerLabel)) / fromIntegral sampleTotal
                           weighted_avg_recall = foldl (+) 0.0 (map (\(pn, recall) -> fromIntegral pn * recall) (zip pn_MultiLabel recallPerLabel)) / fromIntegral sampleTotal
                           weighted_avg_f1Score = foldl (+) 0.0 (map (\(pn, f1Score) -> fromIntegral pn * f1Score) (zip pn_MultiLabel f1ScorePerLabel)) / fromIntegral sampleTotal

                           f1Scores = map (\(pAct, tAct) ->
                                               let tp = length $ intersect pAct tAct
                                                   fp = length $ pAct \\ tAct
                                                   fn = length $ tAct \\ pAct
                                                   prec = case tp + fp > 0 of
                                                            True -> (fromIntegral tp :: Double) / (fromIntegral (tp + fp) :: Double)
                                                            False -> 1.0
                                                   recall = case length tAct > 0 of
                                                              True -> (fromIntegral tp :: Double) / (fromIntegral (tp + fn) :: Double)
                                                              False -> 1.0
                                                   f1Score = 2.0 * prec * recall / (prec + recall)
                                               in f1Score
                                          ) predPairsAct
                           example_f1 = sum f1Scores / (fromIntegral sampleTotal :: Double)

                       putStrLn "\n====【测试集上评估结果】===="
                       putStrLn "注：真正例（TP）、假正例（FP）都没有时，精确率按 1.0 计；样本s的标签集为空时，样本级召回率（Rs)按 1.0 计。"
                       putStrLn $ "测试集样本总数：" ++ show sampleTotal
                       putStrLn $ "      平均损失：" ++ show totalLoss

                       putStrLn "\n====【分类别的完整评估指标】===="
                       putStrLn $ "  [(pAct, tAct)]: " ++ show predPairsAct
                       let correctRate = (fromIntegral (length (filter (\(p,t) -> p == t) predPairsAct)) :: Double) / (fromIntegral sampleTotal :: Double)

                       putStrLn $ "  Correct rate (pAct == tAct) = " ++ (printf "%.4f" correctRate)

                       putStrLn $ "  tpTotal = " ++ show tpTotal ++ ", fpTotal = " ++ show fpTotal ++ ", fnTotal = " ++ show fnTotal
                       putStrLn $ "  f1Scores: " ++ show f1Scores

                       putStrLn $ "  Macro average: Precision = " ++ (printf "%.4f" macro_avg_precision) ++ ", Recall = " ++ (printf "%.4f" macro_avg_recall) ++ ", f1Score: " ++ (printf "%.4f" macro_avg_f1Score)
                       putStrLn $ "  Micro average: Precision = " ++ (printf "%.4f" micro_avg_precision) ++ ", Recall = " ++ (printf "%.4f" micro_avg_recall) ++ ", f1Score: " ++ (printf "%.4f" micro_avg_f1Score)
                       putStrLn $ "  weighted avg.: Precision = " ++ (printf "%.4f" weighted_avg_precision) ++ ", weighted_avg_recall: " ++ (printf "%.4f" weighted_avg_recall) ++ ", weighted_avg_f1Score: " ++ (printf "%.4f" weighted_avg_f1Score)

                       putStrLn $ "  样本级F1分数：" ++ (printf "%.4f" example_f1)
                 else putStrLn "Cancelled."
    else putStrLn "Cancelled."
