{-# LANGUAGE NamedFieldPuns, RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict #-}

-- Binary Tree-LSTM 句法消岐模型（Tai et al. 2015 Child-Sum 标准实现）
module BiTreeLSTM (
    buildSampleSet,
    testPipeline,
    testPipeline2
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
import Data.Map.Strict as Map hiding (map, foldl, size, filter, splitAt, foldl')
import Debug.Trace

import System.Random (randomRIO, mkStdGen, randomR, StdGen, newStdGen)
import List.Shuffle (shuffle_)
import Data.Maybe (fromMaybe)
import Text.Printf (printf)
import Data.List (foldl')
import Data.Tuple.Utils (fst3, snd3, thd3)
import Control.Monad (when, foldM)
import GHC.Generics (Generic)
import Control.Exception (SomeException, bracket, catch)
import System.IO (readFile, writeFile)
import Data.Aeson (encode, decode)
import Database.MySQL.Base

import Utils hiding (isLeafNode)
import Category (Category(..))
import AmbiResol (Prior(..), PhraSyn0, stringToCTPList, priorWithHighestFreq, getPhraSyn0FromStr)
import Phrase (Tag, PhraStru, Seman)
import CL
import Statistics (cate2Vec, tag2Vec, stru2Vec)
import Database
import Rule (seman2SynSeman)

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

-- ===================== 目标动作 One-Hot 转换 =====================
actionToOneHot :: Prior -> Vector Double
actionToOneHot Lp   = LA.fromList [1.0, 0.0, 0.0]
actionToOneHot Rp   = LA.fromList [0.0, 1.0, 0.0]
actionToOneHot Noth = LA.fromList [0.0, 0.0, 1.0]

vecToAction :: Vector Double -> Prior
vecToAction v
  | LA.size v /= 3 = Noth               -- Easy to processing
  | otherwise = case LA.maxIndex v of
      0 -> Lp
      1 -> Rp
      2 -> Noth

-- ===================== 语法树定义 =====================
type NodeFeat = Vector Double            -- Embedding vector, [Double], of a word or a phrase.
data SyntaxTree a = EmptySTree
                  | STreeNode
                      { nodeData :: a
                      , stLeft   :: SyntaxTree a
                      , stRight  :: SyntaxTree a
                      , stIsLeaf :: Bool
                      } deriving (Show, Eq)
type RawSyntaxTree = SyntaxTree NodeFeat

-- 训练样本：((左树,右树),目标动作)
type TrainSample = ((RawSyntaxTree, RawSyntaxTree), Prior)

-- JSON 样本包装结构（如需持久化，可自行扩展 Vector JSON 实例）
data SampleRecord = SampleRecord
                    { recLeftTree  :: RawSyntaxTree
                    , recRightTree :: RawSyntaxTree
                    , recTargetAct :: Prior
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

-- 损失的交叉熵 -Σ l*log(p)
crossEntropy :: Vector Double -> Vector Double -> Double
crossEntropy pred label
  | LA.size pred /= LA.size label = error "crossEntropy: pred/label dimension mismatch"
  | otherwise = - sumElements (LAD.zipVectorWith (\p l -> l * log (max p 1e-8)) pred label)

-- ===================== 网络参数 & 梯度结构 =====================
data NetParam = NetParam
  { -- ===== 新增：句法范畴向量MLP（8 → cateHid1Dim → cateMLPOutDim）=====
    cateMLPW1 :: Matrix Double
  , cateMLPB1 :: Vector Double
  , cateMLPW2 :: Matrix Double
  , cateMLPB2 :: Vector Double
    -- ===== 新增：词嵌入向量MLP（300 → wordHid1Dim → wordMLPOutDim）=====
  , wordMLPW1 :: Matrix Double
  , wordMLPB1 :: Vector Double
  , wordMLPW2 :: Matrix Double
  , wordMLPB2 :: Vector Double
    -- ========== 原有参数不变 ==========
  , leafProjW    :: Matrix Double
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
  { ncCateVec :: Vector Double       -- 新增MLP缓存，仅叶子节点有效
  , ncCateH1Pre :: Vector Double
  , ncCateH1 :: Vector Double
  , ncCateMLPOut :: Vector Double
  , ncWordVec :: Vector Double
  , ncWordH1Pre :: Vector Double
  , ncWordH1 :: Vector Double
  , ncWordMLPOut :: Vector Double
    -- 下面是原有的
  , ncFeat      :: NodeFeat          -- 节点的特征向量, {cateMLPOut; wordMLPOut} 或 {cateVec; tagVec; struVec}
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
 - 输入的原始特征，其实是 MLP 的输出向量。
 - 对于叶子，MLP 输出 {cateMLPOut;wordMLPOut}；对于非叶子，MLP 输出 {cateVec;tagVec;struVec}。
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
initNetParam :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO NetParam
initNetParam cateRawDim wordRawDim cateHid1 wordHid1 cateMLPOutDim wordMLPOutDim nonLeafRawDim projDim hidDim = do
  let
    g0 = mkStdGen 42                      -- 生成随机函数的种子
    -- ========== 新增：初始化cate MLP 8 → cateHid1Dim → cateMLPOutDim ==========
    (cateW1, g1) = randMat cateHid1 cateRawDim g0
    (cateB1, g2) = randVec cateHid1 g1
    (cateW2, g3) = randMat cateMLPOutDim cateHid1 g2
    (cateB2, g4) = randVec cateMLPOutDim g3
    -- ========== 新增：初始化word MLP 300 → wordHid1Dim → wordMLPOutDim ==========
    (wordW1, g5) = randMat wordHid1 wordRawDim g4
    (wordB1, g6) = randVec wordHid1 g5
    (wordW2, g7) = randMat wordMLPOutDim wordHid1 g6
    (wordB2, g8) = randVec wordMLPOutDim g7
    -- ========== 原有参数初始化，g从g8继续 ==========
    leafMLPDim = cateMLPOutDim + wordMLPOutDim          -- 仿射投影前，叶子特征向量（MLP投影后的向量的拼接）的维度
    (lProjW, g9) = randMat projDim leafMLPDim g8
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

    (clsW, g33) = randMat 3 (2 * hidDim) g32    -- 两棵树的根节点的隐状态拼接到三分类的投影矩阵
    (clsB, _) = randVec 3 g33                   -- 三分类投影的偏置向量

  return NetParam
    { cateMLPW1 = cateW1, cateMLPB1 = cateB1    -- 1st hidden layer for category embedding vector
    , cateMLPW2 = cateW2, cateMLPB2 = cateB2    -- 2nd hidden layer for category embedding vector
    , wordMLPW1 = wordW1, wordMLPB1 = wordB1    -- 1st hidden layer for word embedding vector
    , wordMLPW2 = wordW2, wordMLPB2 = wordB2    -- 2nd hidden layer for word embedding vector
    , leafProjW = lProjW, leafProjB = lProjB    -- Project layer for leaf node
    , nonLeafProjW = nlProjW, nonLeafProjB = nlProjB      -- Project layer for non-leaf node
    , inpW = iW, inpLW = iLW, inpRW = iRW, inpB = iB      -- Input gate
    , forgetLW = fLW, forgetLLW = fLLW, forgetLRW = fLRW, forgetLB = fLB      -- Left forget gate
    , forgetRW = fRW, forgetRLW = fRLW, forgetRRW = fRRW, forgetRB = fRB      -- Right forget gate
    , outW = oW, outLW = oLW, outRW = oRW, outB = oB                          -- Output gate
    , cellCandW = cW, cellCandLW = cLW, cellCandRW = cRW, cellCandB = cB      -- Cell state
    , clsWeight = clsW, clsBias = clsB                                        -- Classification layer
    }

-- 清零网络参数的梯度，即损失 Loss 对每个参数的偏导数都是 0.0
zeroGrad :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> NetGrad
zeroGrad cateInDim wordInDim cateHid1Dim wordHid1Dim cateMLPOutDim wordMLPOutDim nonLeafRawDim projDim hidDim =
  let zm r c = konst 0.0 (r, c)
      zv d = konst 0.0 d
  in NetParam
    { -- category MLP grad zero
      cateMLPW1 = zm cateHid1Dim cateInDim
    , cateMLPB1 = zv cateHid1Dim
    , cateMLPW2 = zm cateMLPOutDim cateHid1Dim
    , cateMLPB2 = zv cateMLPOutDim
      -- word MLP grad zero
    , wordMLPW1 = zm wordHid1Dim wordInDim
    , wordMLPB1 = zv wordHid1Dim
    , wordMLPW2 = zm wordMLPOutDim wordHid1Dim
    , wordMLPB2 = zv wordMLPOutDim
      -- Project layer grad zero
    , leafProjW = zm projDim (cateMLPOutDim + wordMLPOutDim), leafProjB = zv projDim
    , nonLeafProjW = zm projDim nonLeafRawDim, nonLeafProjB = zv projDim
      -- LSTM basic unit grad zero
    , inpW  = zm hidDim projDim, inpLW = zm hidDim hidDim, inpRW = zm hidDim hidDim, inpB = zv hidDim
    , forgetLW = zm hidDim projDim, forgetLLW = zm hidDim hidDim, forgetLRW = zm hidDim hidDim, forgetLB = zv hidDim
    , forgetRW = zm hidDim projDim, forgetRLW = zm hidDim hidDim, forgetRRW = zm hidDim hidDim, forgetRB = zv hidDim
    , outW  = zm hidDim projDim, outLW = zm hidDim hidDim, outRW = zm hidDim hidDim, outB = zv hidDim
    , cellCandW = zm hidDim projDim, cellCandLW = zm hidDim hidDim, cellCandRW = zm hidDim hidDim, cellCandB = zv hidDim
    , clsWeight = zm 3 (2 * hidDim), clsBias = zv 3
    }

-- ================= 梯度的辅助函数 =================
-- 网络参数的梯度加
gradAdd :: NetGrad -> NetGrad -> NetGrad
gradAdd g1 g2 = NetParam
    { cateMLPW1 = cateMLPW1 g1 + cateMLPW1 g2         -- MLP gradient addition
    , cateMLPB1 = cateMLPB1 g1 + cateMLPB1 g2
    , cateMLPW2 = cateMLPW2 g1 + cateMLPW2 g2
    , cateMLPB2 = cateMLPB2 g1 + cateMLPB2 g2
    , wordMLPW1 = wordMLPW1 g1 + wordMLPW1 g2
    , wordMLPB1 = wordMLPB1 g1 + wordMLPB1 g2
    , wordMLPW2 = wordMLPW2 g1 + wordMLPW2 g2
    , wordMLPB2 = wordMLPB2 g1 + wordMLPB2 g2
    , leafProjW = leafProjW g1 + leafProjW g2         -- Projection gradient addition
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
    { cateMLPW1 = scale s (cateMLPW1 g1)
    , cateMLPB1 = scale s (cateMLPB1 g1)
    , cateMLPW2 = scale s (cateMLPW2 g1)
    , cateMLPB2 = scale s (cateMLPB2 g1)
    , wordMLPW1 = scale s (wordMLPW1 g1)
    , wordMLPB1 = scale s (wordMLPB1 g1)
    , wordMLPW2 = scale s (wordMLPW2 g1)
    , wordMLPB2 = scale s (wordMLPB2 g1)
    , leafProjW = scale s (leafProjW g1)
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
      collectNorms = [ matNorm (cateMLPW1 g), vecNorm (cateMLPB1 g)
                     , matNorm (cateMLPW2 g), vecNorm (cateMLPB2 g)
                     , matNorm (wordMLPW1 g), vecNorm (wordMLPB1 g)
                     , matNorm (wordMLPW2 g), vecNorm (wordMLPB2 g)
                     , matNorm (leafProjW g), vecNorm (leafProjB g)
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
    { cateMLPW1 = updateMat (cateMLPW1 p) (cateMLPW1 grad)            -- Update gradient of MLP
    , cateMLPB1 = updateVec (cateMLPB1 p) (cateMLPB1 grad)
    , cateMLPW2 = updateMat (cateMLPW2 p) (cateMLPW2 grad)
    , cateMLPB2 = updateVec (cateMLPB2 p) (cateMLPB2 grad)
    , wordMLPW1 = updateMat (wordMLPW1 p) (wordMLPW1 grad)
    , wordMLPB1 = updateVec (wordMLPB1 p) (wordMLPB1 grad)
    , wordMLPW2 = updateMat (wordMLPW2 p) (wordMLPW2 grad)
    , wordMLPB2 = updateVec (wordMLPB2 p) (wordMLPB2 grad)
    , leafProjW    = updateMat (leafProjW p) (leafProjW grad)         -- Update gradient of projection layer
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

-- Cate MLP前向：cateVec(8) -> HideLayer(dim_cate_hide_layer) -> cateVec'(dim_cate_mlpout), using activation function sigmoid.
mlpCateForward :: NetParam -> Vector Double -> (Vector Double, Vector Double, Vector Double)
mlpCateForward NetParam{cateMLPW1, cateMLPB1, cateMLPW2, cateMLPB2} x =
  let h1Pre = cateMLPW1 #> x + cateMLPB1
      h1 = vecSigmoid h1Pre
      out = cateMLPW2 #> h1 + cateMLPB2
  in (out, h1, h1Pre)                      -- 返回输出、第一层激活、第一层预激活

-- Word MLP前向：wordVec(300) -> HideLayer(dim_word_hide_layer) -> wordVec'(dim_word_mlpout)，using activation function sigmoid.
mlpWordForward :: NetParam -> Vector Double -> (Vector Double, Vector Double, Vector Double)
mlpWordForward NetParam{wordMLPW1, wordMLPB1, wordMLPW2, wordMLPB2} x =
  let h1Pre = wordMLPW1 #> x + wordMLPB1
      h1 = vecSigmoid h1Pre
      out = wordMLPW2 #> h1 + wordMLPB2
  in (out, h1, h1Pre)                      -- 返回输出、第一层激活、第一层预激活

{- MLP 反向函数
 - 签名：mlpCateBackward param grad dLdOut out h1 h1Pre x -> (updatedGrad, dLdX)
 -}
mlpCateBackward :: NetParam -> NetGrad -> Vector Double -> Vector Double -> Vector Double -> Vector Double -> Vector Double -> (NetGrad, Vector Double)
mlpCateBackward NetParam{cateMLPW1,cateMLPB1,cateMLPW2,cateMLPB2} netGrad dLdOut out h1 h1Pre x =
   let
     dLdH1 = tr' cateMLPW2 #> dLdOut
     dLdH1Pre = dLdH1 * sigmoidDeriv h1Pre
     dLdX = tr' cateMLPW1 #> dLdH1Pre
     gradW2 = outer dLdOut h1
     gradB2 = dLdOut
     gradW1 = outer dLdH1Pre x
     gradB1 = dLdH1Pre
     -- 只修改传入grad网络的cate部分，其余保留
     mlpGrad = netGrad
       { cateMLPW1 = gradW1
       , cateMLPB1 = gradB1
       , cateMLPW2 = gradW2
       , cateMLPB2 = gradB2
       }
   in (mlpGrad, dLdX)

mlpWordBackward :: NetParam -> NetGrad -> Vector Double -> Vector Double -> Vector Double -> Vector Double -> Vector Double -> (NetGrad, Vector Double)
mlpWordBackward NetParam{wordMLPW1,wordMLPB1,wordMLPW2,wordMLPB2} netGrad dLdOut out h1 h1Pre x =
   let
     dLdH1 = tr' wordMLPW2 #> dLdOut
     dLdH1Pre = dLdH1 * sigmoidDeriv h1Pre
     dLdX = tr' wordMLPW1 #> dLdH1Pre
     gradW2 = outer dLdOut h1
     gradB2 = dLdOut
     gradW1 = outer dLdH1Pre x
     gradB1 = dLdH1Pre
     mlpGrad = netGrad
       { wordMLPW1 = gradW1
       , wordMLPB1 = gradB1
       , wordMLPW2 = gradW2
       , wordMLPB2 = gradB2
       }
   in (mlpGrad, dLdX)

{- ===================== Binary Tree-LSTM 前向传播 =====================
 - Computing output of current node. Input 'feat' is xw or xp, here xw = {cateVec;embVec}, and xp = {cateVec;tagVec;struVec}.
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
    hidDim = netHidDim param
    dim_cate = snd $ LA.size cateMLPW1                 -- Usually, it is 8.
    dim_word = snd $ LA.size wordMLPW1                 -- Usually, it is 300.

    -- ========== 新增叶子MLP分支 ==========
    (cateVec, wordVec, cateMLPOut, cateH1, cateH1Pre, wordMLPOut, wordH1, wordH1Pre, rawFeatAfterMLP)
      | isLeafFlag =
          let
            cateRaw = subVector 0 dim_cate feat               -- 前8维 cateRaw
            wordRaw = subVector dim_cate dim_word feat        -- 后300维 wordRaw
            (cateOut, cateH1, cateH1Pre) = mlpCateForward param cateRaw
            (wordOut, wordH1, wordH1Pre) = mlpWordForward param wordRaw
            combinedMLPFeat = LA.vjoin [cateOut, wordOut]
          in (cateRaw, wordRaw, cateOut, cateH1, cateH1Pre, wordOut, wordH1, wordH1Pre, combinedMLPFeat)
      | otherwise =
          -- 非叶子节点：保留原有feat，MLP缓存填零向量
          let zero8 = konst 0 dim_cate
              zero300 = konst 0 dim_word
              zeroCateH = konst 0 (LA.size cateMLPB1)
              zeroWordH = konst 0 (LA.size wordMLPB1)
              zeroCateOut = konst 0 (LA.size cateMLPB2)
              zeroWordOut = konst 0 (LA.size wordMLPB2)
          in (zero8, zero300, zeroCateOut, zeroCateH, zeroCateH, zeroWordOut, zeroWordH, zeroWordH, feat)

    projFeat = projectRawFeat param isLeafFlag rawFeatAfterMLP
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
      { ncCateVec = cateVec
      , ncCateH1Pre = cateH1Pre
      , ncCateH1 = cateH1
      , ncCateMLPOut = cateMLPOut
      , ncWordVec = wordVec
      , ncWordH1Pre = wordH1Pre
      , ncWordH1 = wordH1
      , ncWordMLPOut = wordMLPOut
      , ncFeat = rawFeatAfterMLP
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

    dLdMLPOutFeat = case ncIsLeaf of
                      True -> tr' leafProjW #> dLdProjFeat      -- dL/d [cateMLPout; wordMLPout]
                      False -> tr' nonLeafProjW #> dLdProjFeat  -- dL/d [cateVec; tagVec; struVec]

    dim_cate = snd $ LA.size cateMLPW1        -- dim_cate, usually it is 8.
    dim_word = snd $ LA.size wordMLPW1        -- dim_word, usually it is 300.
    cateHid1Dim = LA.size cateMLPB1
    wordHid1Dim = LA.size wordMLPB1
    cateMLPOutDim = LA.size cateMLPB2
    wordMLPOutDim = LA.size wordMLPB2
    nonLeafRawDim = snd $ LA.size nonLeafProjW
    projDim = LA.size leafProjB
    hidDim = LA.size inpB
    zeroG = zeroGrad dim_cate dim_word cateHid1Dim wordHid1Dim cateMLPOutDim wordMLPOutDim nonLeafRawDim projDim hidDim      -- Zero gradient network.

    (projGrad, dLdRawFeat)
      | ncIsLeaf =
          let
              -- dLdMLPOutFeat: dL/d [cateMLPout; wordMLPout]
              debugMsg = "cateMLPOutDim = " ++ show cateMLPOutDim ++ ", wordMLPOutDim = " ++ show wordMLPOutDim ++ ", dLdMLPOutFeat = " ++ show (LA.size dLdMLPOutFeat)
              dLdCateMLPOut = traceIO debugMsg `seq` subVector 0 cateMLPOutDim dLdMLPOutFeat
              dLdWordMLPOut = subVector cateMLPOutDim wordMLPOutDim dLdMLPOutFeat
              -- 取出缓存在NodeCache里的MLP中间值
              (cateMLPGrad, dLdCateIn) = mlpCateBackward param zeroG dLdCateMLPOut ncCateMLPOut ncCateH1 ncCateH1Pre ncCateVec
              (wordMLPGrad, dLdWordIn) = mlpWordBackward param zeroG dLdWordMLPOut ncWordMLPOut ncWordH1 ncWordH1Pre ncWordVec
              dLdRawFeatSplit = vjoin [dLdCateIn, dLdWordIn]

              gW = outer dLdProjFeat ncFeat            -- rawFeatAfterMLP = {ncCateMLPOut; ncWordMLPOut}
              gB = dLdProjFeat
              pg = zeroG { leafProjW = gW, leafProjB = gB }                     -- Projection layer gradient
              fullLeafGrad = pg `gradAdd` cateMLPGrad `gradAdd` wordMLPGrad
          in (fullLeafGrad, dLdRawFeatSplit)
      | otherwise =
          let gW = outer dLdProjFeat ncFeat            -- rawFeatAfterMLP = {cateVec; tagVec; struVec}
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
    dim_cate = snd $ LA.size (cateMLPW1 param)          -- Usually it is 8.
    dim_word = snd $ LA.size (wordMLPW1 param)          -- Usually it is 300.
    cateHid1Dim = fst $ LA.size (cateMLPW1 param)
    wordHid1Dim = fst $ LA.size (wordMLPW1 param)
    cateMLPOutDim = fst $ LA.size (cateMLPW2 param)
    wordMLPOutDim = fst $ LA.size (wordMLPW2 param)
    nonLeafRawDim = snd $ LA.size (nonLeafProjW param)
    projDim = fst $ LA.size (leafProjW param)
    hidDim = fst $ LA.size (inpW param)
  in zeroGrad dim_cate dim_word cateHid1Dim wordHid1Dim cateMLPOutDim wordMLPOutDim nonLeafRawDim projDim hidDim
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

    concatH = LA.vjoin [hl, hr]
    NetParam{..} = param
    logits = clsWeight #> concatH + clsBias
    pred = softmax logits
    label = actionToOneHot action
    loss = crossEntropy pred label
    dLdLogit = pred - label                   -- dLdz
    gradClsW = outer dLdLogit concatH
    gradClsB = dLdLogit
    hidDim = LA.size hl
    dLdConcatH = tr' clsWeight #> dLdLogit
    dLdHl = subVector 0 hidDim dLdConcatH
    dLdHr = subVector hidDim (LA.size dLdConcatH - hidDim) dLdConcatH
    gradL = backwardTree param (tfCacheTree lFwd) dLdHl (konst 0.0 (size dLdHl))
    gradR = backwardTree param (tfCacheTree rFwd) dLdHr (konst 0.0 (size dLdHr))
    totalGrad0 = gradL `gradAdd` gradR
    totalGrad = totalGrad0
      { clsWeight = gradClsW
      , clsBias = gradClsB
      }
  in (loss, totalGrad)

batchTrainStep :: NetParam -> [TrainSample] -> Double -> (Double, NetParam)
batchTrainStep param samples lr =
  let
    dim_cate = snd $ LA.size (cateMLPW1 param)          -- Usually it is 8.
    dim_word = snd $ LA.size (wordMLPW1 param)          -- Usually it is 300.
    cateHid1Dim = LA.size $ cateMLPB1 param
    wordHid1Dim = LA.size $ wordMLPB1 param
    cateMLPOutDim = LA.size $ cateMLPB2 param
    wordMLPOutDim = LA.size $ wordMLPB2 param
    nonLeafRawDim = snd $ LA.size $ nonLeafProjW param
    projDim = netProjDim param
    hidDim = netHidDim param
    initG = zeroGrad dim_cate dim_word cateHid1Dim wordHid1Dim cateMLPOutDim wordMLPOutDim nonLeafRawDim projDim hidDim
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
 - type PhraSyn0 = (Category, Tag, PhraStru)
 - Count OOV words and non-OOV words. When embedding vectors of words are not used, oovCnt and noovCnt denote number of leaves and that of non-leaves.
 -}
biTreePhraSyn0Seman2SyntaxTreeNodeFeat :: BiTree (PhraSyn0, Term)
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
biTreePhraSyn0Seman2SyntaxTreeNodeFeat Empty _ _ _ _ _ _ _ _ _ _ = return (EmptySTree, 0, 0)
biTreePhraSyn0Seman2SyntaxTreeNodeFeat (Node (phraSyn0, semanTerm) lt rt) cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov oovCnt noovCnt useEmbVec = do
--    putStrLn $ "PhraSyn0: (" ++ show (fst3 phraSyn0) ++ ", " ++ throwFstLast (snd3 phraSyn0) ++ ", " ++ throwFstLast (thd3 phraSyn0) ++ ")"
    let
        (v1, g1) = randVec dim_cate (mkStdGen 42)
        (v2, g2) = randVec dim_tag g1
        (v3, _) = randVec dim_stru g2
        cate = fst3 phraSyn0
        tag = (throwFstLast . snd3) phraSyn0                 -- String as "<" and "T->B" in which there is no double quotes.
        stru = (throwFstLast . thd3) phraSyn0                -- String as "AHn" and "OE" in which there is no double quotes.
        cateVec' = Map.lookup cate cateVecMap                -- Maybe Double
        tagVec' = Map.lookup tag tagVecMap
        struVec' = Map.lookup stru struVecMap
        cateVec = fromMaybe v1 cateVec'
        tagVec = fromMaybe v2 tagVec'
        struVec = fromMaybe v3 struVec'

{-
    case cateVec' of
      Just x -> putStrLn $ "Find (" ++ show (fst3 phraSyn0) ++ ", " ++ listToString (formatDoubleList (LA.toList x) 4) ++ ")"
      Nothing -> putStrLn $ "Not find vector for category '" ++ show (fst3 phraSyn0) ++ "'"
    case tagVec' of
      Just x -> putStrLn $ "Find (\"" ++ tag ++ "\", " ++ listToString (formatDoubleList (LA.toList x) 4) ++ ")"
      Nothing -> putStrLn $ "Not find vector for tag '" ++ tag ++ "'"          -- When Tag value is 'Desig'
    case struVec' of
      Just x -> putStrLn $ "Find (\"" ++ stru ++ "\", " ++ listToString (formatDoubleList (LA.toList x) 4) ++ ")"
      Nothing -> putStrLn $ "Not find vector for phrasal structure '" ++ stru ++ "'"    -- When PhraStru value is 'DE'

    putStrLn $ "  seman: " ++ show semanTerm
    putStrLn $ "  cateVec: " ++ show cateVec
    putStrLn $ "  tagVec: " ++ show tagVec
    putStrLn $ "  struVec: " ++ show struVec
 -}
    if useEmbVec                 -- Using embedding vector when node is a leaf.
      then
        if isPrimTerm semanTerm
          then do                 -- Word seman
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
            (lt, oovCnt', noovCnt') <- biTreePhraSyn0Seman2SyntaxTreeNodeFeat lt cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov oovCnt noovCnt useEmbVec
            (rt, oovCnt'', noovCnt'') <- biTreePhraSyn0Seman2SyntaxTreeNodeFeat rt cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov oovCnt' noovCnt' useEmbVec
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
            (lt, oovCnt', noovCnt') <- biTreePhraSyn0Seman2SyntaxTreeNodeFeat lt cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov oovCnt noovCnt useEmbVec
            (rt, oovCnt'', noovCnt'') <- biTreePhraSyn0Seman2SyntaxTreeNodeFeat rt cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov oovCnt' noovCnt' useEmbVec
            return ( STreeNode { nodeData = LA.vjoin [cateVec, tagVec, struVec]
                               , stLeft = lt
                               , stRight = rt
                               , stIsLeaf = False
                               }
                   , oovCnt'', noovCnt'' + 1
                   )

buildSampleSet :: Int -> Int -> Int -> IO [SampleRecord]
buildSampleSet dim_cate dim_tag dim_stru = do
  confInfo <- readFile "Configuration"
  let syntax_ambig_resol_model = getConfProperty "syntax_ambig_resol_model" confInfo
      syntax_resol_sample_startId = getConfProperty "syntax_resol_sample_startId" confInfo
      syntax_resol_sample_endId = getConfProperty "syntax_resol_sample_endId" confInfo
      startId = read syntax_resol_sample_startId :: Int
      endId = read syntax_resol_sample_endId :: Int
      leafUsingEmbVec = getConfProperty "leafUsingEmbVec" confInfo
      useEmbVec = read leafUsingEmbVec :: Bool

  conn <- getConn
  let sqlstat = DS.fromString $ "select leftOverTree, leftOverSeman, rightOverTree, rightOverSeman, clauTagPrior from "
                                ++ syntax_ambig_resol_model ++ " where id >= ? && id <= ?"
  stmt <- prepareStmt conn sqlstat
  (defs, is) <- queryStmt conn stmt [toMySQLInt32U startId, toMySQLInt32U endId]
  textTextTextTextTextList <- readStreamByTextTextTextTextText [] is        -- [(String, String, String, String, String)]
  closeStmt conn stmt
  close conn

  let lotLosRotRosPriorList = map (\x -> ( stringToBiTree getPhraSyn0FromStr (fst5 x)           -- BiTree PhraSyn0
                                         , (biTreeEqualsWithTerm . getTermFromStr . seman2SynSeman) (snd5 x)      -- BiTree Term
                                         , stringToBiTree getPhraSyn0FromStr (thd5 x)           -- BiTree PhraSyn0
                                         , (biTreeEqualsWithTerm . getTermFromStr . seman2SynSeman) (fth5 x)      -- BiTree Term
                                         , (fromMaybe Noth . priorWithHighestFreq . stringToCTPList) (fif5 x)     -- Prior
                                         )
                                  ) textTextTextTextTextList

--  putStrLn $ "lotLosRotRosPriorList: " ++ show lotLosRotRosPriorList
  let loRoPriorList = map (\x -> (( mergeBiTree (fst5 x) (snd5 x)                -- BiTree (PhraSyn0, Term)
                                  , mergeBiTree (thd5 x) (fth5 x)
                                  )
                                  , fif5 x
                                 )
                          ) lotLosRotRosPriorList           -- [((BiTree (PhraSyn0, Term), BiTree (PhraSyn0, Term)), Prior)]
--  putStrLn $ "Sample number = " ++ show (length lotLosRotRosPriorList) ++ ", the first is: " ++ show (loRoPriorList!!0)

  cateVecMap <- cate2Vec                         -- Map Category (Vector Double)
  tagVecMap <- tag2Vec                           -- Map Tag (Vector Double)
  struVecMap <- stru2Vec                         -- Map PhraStru (Vector Double)

  sampleSetWithOOVCnt <- mapWithProgressStep 100
      (\x -> do
          (lot, loovCnt, lnoovCnt) <- biTreePhraSyn0Seman2SyntaxTreeNodeFeat ((fst . fst) x) cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov300 0 0 useEmbVec
          (rot, roovCnt, rnoovCnt) <- biTreePhraSyn0Seman2SyntaxTreeNodeFeat ((snd . fst) x) cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru oov300 0 0 useEmbVec
          let prior = snd x
          return ( SampleRecord{ recLeftTree = lot
                               , recRightTree = rot
                               , recTargetAct = prior
                               }
                  , loovCnt + roovCnt                  -- Total of OOV words in two trees
                  , lnoovCnt + rnoovCnt                -- Total of non-OOV words in two trees
                  )
      ) loRoPriorList
  let sampleSet = map fst3 sampleSetWithOOVCnt
      oovTotal = foldl (+) 0 $ map snd3 sampleSetWithOOVCnt
      noovTotal = foldl (+) 0 $ map thd3 sampleSetWithOOVCnt
      prop = 100 * (fromIntegral oovTotal / fromIntegral noovTotal) :: Double
--  putStrLn $ "buildSampleSet: sampleSet!!0: " ++ show (sampleSet!!0)
  putStrLn $ "buildSampleSet: oovTotal = " ++ show oovTotal ++ ", noovTotal = " ++ show noovTotal ++ ", proportion of oov words = " ++ (printf "%.2f" prop) ++ "%"
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
          let syntax_ambig_resol_model = getConfProperty "syntax_ambig_resol_model" confInfo
              syntax_resol_sample_startId = getConfProperty "syntax_resol_sample_startId" confInfo
              syntax_resol_sample_endId = getConfProperty "syntax_resol_sample_endId" confInfo
              startId = read syntax_resol_sample_startId :: Int
              endId = read syntax_resol_sample_endId :: Int
              sampleNum = endId - startId + 1
              dim_cate_hide_layer = getConfProperty "dim_cate_hide_layer" confInfo
              dim_cate_mlpout = getConfProperty "dim_cate_mlpout" confInfo
              dim_word_mlpout = getConfProperty "dim_word_mlpout" confInfo
              dim_hidden = getConfProperty "dim_hidden" confInfo

              lr_str = show $ floor (lr * 1000)
              tblNameSuffix = "_mlp_h" ++ dim_cate_hide_layer ++ "_c" ++ dim_cate_mlpout ++ "_w" ++ dim_word_mlpout ++ "_dim" ++ dim_hidden ++ "_lr" ++ lr_str ++ "_ns" ++ show sampleNum
              tblName = syntax_ambig_resol_model ++ tblNameSuffix
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
testPipeline :: IO ()
testPipeline = do
  putStrLn "TODO"

--- ===================== 训练与测试：可独立运行 =====================
testPipeline2 :: IO ()
testPipeline2 = do
  putStrLn "==== Binary Tree-LSTM ===="
  confInfo <- readFile "Configuration"
  let
    syntax_ambig_resol_model = getConfProperty "syntax_ambig_resol_model" confInfo
    syntax_resol_sample_startId = getConfProperty "syntax_resol_sample_startId" confInfo
    syntax_resol_sample_endId = getConfProperty "syntax_resol_sample_endId" confInfo
    startId = read syntax_resol_sample_startId :: Int
    endId = read syntax_resol_sample_endId :: Int

    dim_word_str = getConfProperty "dim_word" confInfo
    dim_word_hide_layer_str = getConfProperty "dim_word_hide_layer" confInfo
    dim_word_mlpout_str = getConfProperty "dim_word_mlpout" confInfo
    dim_cate_str = getConfProperty "dim_cate" confInfo
    dim_cate_hide_layer_str = getConfProperty "dim_cate_hide_layer" confInfo
    dim_cate_mlpout_str = getConfProperty "dim_cate_mlpout" confInfo
    dim_tag_str = getConfProperty "dim_tag" confInfo
    dim_stru_str = getConfProperty "dim_stru" confInfo
    dim_proj_str = getConfProperty "dim_proj" confInfo
    dim_hidden_str = getConfProperty "dim_hidden" confInfo
    lr_str = getConfProperty "lr" confInfo
    epochs_str = getConfProperty "epochs" confInfo
    earlyStopPatience_str = getConfProperty "earlyStopPatience" confInfo
    batchSize_str = getConfProperty "batchSize" confInfo
    useEmbVec_str = getConfProperty "leafUsingEmbVec" confInfo

    dim_word = read dim_word_str :: Int
    dim_word_hide_layer = read dim_word_hide_layer_str :: Int
    dim_word_mlpout = read dim_word_mlpout_str :: Int
    dim_cate = read dim_cate_str :: Int
    dim_cate_hide_layer = read dim_cate_hide_layer_str :: Int
    dim_cate_mlpout = read dim_cate_mlpout_str :: Int
    dim_tag = read dim_tag_str :: Int
    dim_stru = read dim_stru_str :: Int
    leafRawDim = dim_cate + dim_word
    leafMLPDim = dim_word_mlpout + dim_cate_mlpout
    nonLeafRawDim = dim_cate + dim_tag + dim_stru
    projDim = read dim_proj_str :: Int
    hidDim = read dim_hidden_str :: Int
    lr = read lr_str :: Double
    epochs = read epochs_str :: Int
    earlyStopPatience = read earlyStopPatience_str :: Int
    batchSize = read batchSize_str :: Int
    useEmbVec = read useEmbVec_str :: Bool

  putStrLn $ " syntax_ambig_resol_model: " ++ syntax_ambig_resol_model
  putStrLn $ " syntax_resol_sample_startId: " ++ syntax_resol_sample_startId
  putStrLn $ " syntax_resol_sample_endId: " ++ syntax_resol_sample_endId
  putStrLn $ " dim_word: " ++ dim_word_str
  putStrLn $ " dim_word_hide_layer: " ++ dim_word_hide_layer_str
  putStrLn $ " dim_word_mlpout: " ++ dim_word_mlpout_str
  putStrLn $ " dim_cate: " ++ dim_cate_str
  putStrLn $ " dim_cate_hide_layer: " ++ dim_cate_hide_layer_str
  putStrLn $ " dim_cate_mlpout: " ++ dim_cate_mlpout_str
  putStrLn $ " dim_tag: " ++ dim_tag_str
  putStrLn $ " dim_stru: " ++ dim_stru_str
  putStrLn $ " leafRawDim: " ++ show leafRawDim
  putStrLn $ " leafMLPDim: " ++ show leafMLPDim
  putStrLn $ " nonLeafRawDim: " ++ show nonLeafRawDim
  putStrLn $ " projDim: " ++ dim_proj_str
  putStrLn $ " hidDim: " ++ dim_hidden_str
  putStrLn $ " lr: " ++ lr_str
  putStrLn $ " epochs: " ++ epochs_str
  putStrLn $ " earlyStopPatience: " ++ earlyStopPatience_str
  putStrLn $ " batchSize: " ++ batchSize_str
  putStrLn $ " useEmbVec: " ++ useEmbVec_str

  let prompt = " Train Binary tree LSTM for syntactic disambiguity in Module BiTreeLSTM, are you sure? [y/n] (RETURN for 'y'): "
  answer <- getLineUntil prompt ["y","n"] True
  if answer == "y"
    then do
      -- ========== 清空存储训练过程损失率的 MySQL 表 ==========
      let
          lr_str = show $ floor (lr * 1000)
          startId = read syntax_resol_sample_startId :: Int
          endId = read syntax_resol_sample_endId :: Int
          sampleNum = endId - startId + 1
          tblNameSuffix = "_mlp_h" ++ dim_cate_hide_layer_str ++ "_c" ++ dim_cate_mlpout_str ++ "_w" ++ dim_word_mlpout_str ++ "_dim" ++ dim_hidden_str ++ "_lr" ++ lr_str ++ "_ns" ++ show sampleNum
          tblName = syntax_ambig_resol_model ++ tblNameSuffix
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

                   initP <- initNetParam dim_cate dim_word dim_cate_hide_layer dim_word_hide_layer dim_cate_mlpout dim_word_mlpout nonLeafRawDim projDim hidDim

                   samples0 <- buildSampleSet dim_cate dim_tag dim_stru
                   let samples = map (\x -> ((recLeftTree x, recRightTree x), recTargetAct x)) samples0
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
                           predPairs = map (\s -> let ((l,r),t) = s
                                                      fL = treeForwardCached bestTrainedP l
                                                      fR = treeForwardCached bestTrainedP r
                                                      LSTMState hL _ = tfRootState fL
                                                      LSTMState hR _ = tfRootState fR
                                                      concatH = LA.vjoin [hL,hR]
                                                      log = clsWeight bestTrainedP #> concatH + clsBias bestTrainedP
                                                      pAct = vecToAction (softmax log)
                                                  in (pAct, t)
                                           ) testSet
                           correct = length $ filter (\(p,t) -> p == t) predPairs
                           total = length testSet
                           acc = fromIntegral correct / fromIntegral total :: Double

                       -- True/False: Predict correctly, Positive/Negative: Prediction result
                       -- TP: True Positive, FN: False Negative, FP: False Positive.

                           tp_Lp = length $ filter (\(p,t) -> p == Lp && t == Lp) predPairs   -- True Positive for 'Lp'
                           tp_Rp = length $ filter (\(p,t) -> p == Rp && t == Rp) predPairs   -- True Positive for 'Rp'
                           tp_Noth = length $ filter (\(p,t) -> p == Noth && t == Noth) predPairs   -- True Positive for 'Noth'

                           fn_Lp = length $ filter (\(p,t) -> p /= Lp && t == Lp) predPairs   -- False Negative for 'Lp'
                           fn_Rp = length $ filter (\(p,t) -> p /= Rp && t == Rp) predPairs   -- False Negative for 'Rp'
                           fn_Noth = length $ filter (\(p,t) -> p /= Noth && t == Noth) predPairs   -- False Negative for 'Noth'

                           fp_Lp = length $ filter (\(p,t) -> p == Lp && t /= Lp) predPairs   -- False Positives for 'Lp'
                           fp_Rp = length $ filter (\(p,t) -> p == Rp && t /= Rp) predPairs   -- False Positives for 'Rp'
                           fp_Noth = length $ filter (\(p,t) -> p == Noth && t /= Noth) predPairs   -- False Positives for 'Noth'

                           precision_Lp = case tp_Lp + fp_Lp > 0 of
                                            True -> fromIntegral tp_Lp / (fromIntegral (tp_Lp + fp_Lp))
                                            False -> 0.0
                           precision_Rp = case tp_Rp + fp_Rp > 0 of
                                            True -> fromIntegral tp_Rp / (fromIntegral (tp_Rp + fp_Rp))
                                            False -> 0.0
                           precision_Noth = case tp_Noth + fp_Noth > 0 of
                                              True -> fromIntegral tp_Noth / (fromIntegral (tp_Noth + fp_Noth))
                                              False -> 0.0

                           recall_Lp = case tp_Lp + fn_Lp > 0 of
                                         True -> fromIntegral tp_Lp / (fromIntegral (tp_Lp + fn_Lp))
                                         False -> 0.0
                           recall_Rp = case tp_Rp + fn_Rp > 0 of
                                         True -> fromIntegral tp_Rp / (fromIntegral (tp_Rp + fn_Rp))
                                         False -> 0.0
                           recall_Noth = case tp_Noth + fn_Noth > 0 of
                                           True -> fromIntegral tp_Noth / (fromIntegral (tp_Noth + fn_Noth))
                                           False -> 0.0

                           f1Score_Lp = case precision_Lp + recall_Lp > 0 of
                                          True -> 2 * precision_Lp * recall_Lp / (precision_Lp + recall_Lp)
                                          False -> 0.0
                           f1Score_Rp = case precision_Rp + recall_Rp > 0 of
                                          True -> 2 * precision_Rp * recall_Rp / (precision_Rp + recall_Rp)
                                          False -> 0.0
                           f1Score_Noth = case precision_Noth + recall_Noth > 0 of
                                            True -> 2 * precision_Noth * recall_Noth / (precision_Noth + recall_Noth)
                                            False -> 0.0

                           macro_avg_precision = (precision_Lp + precision_Rp + precision_Noth) / 3
                           macro_avg_recall = (recall_Lp + recall_Rp + recall_Noth) / 3
                           macro_avg_f1Score = (f1Score_Lp + f1Score_Rp + f1Score_Noth) / 3

                           n_Lp = fromIntegral (tp_Lp + fn_Lp)
                           n_Rp = fromIntegral (tp_Rp + fn_Rp)
                           n_Noth = fromIntegral (tp_Noth + fn_Noth)

                           weighted_avg_precision = (n_Lp * precision_Lp + n_Rp * precision_Rp + n_Noth * precision_Noth) / (fromIntegral total :: Double)
                           weighted_avg_recall = (n_Lp * recall_Lp + n_Rp * recall_Rp + n_Noth * recall_Noth) / (fromIntegral total :: Double)
                           weighted_avg_f1Score = (n_Lp * f1Score_Lp + n_Rp * f1Score_Rp + n_Noth * f1Score_Noth) / (fromIntegral total :: Double)

                       putStrLn "\n====【完整测试集正式评估结果】===="
                       putStrLn $ "测试集平均损失：" ++ show totalLoss
                       putStrLn $ "正确样本数：" ++ show correct ++ " / " ++ show total
                       putStrLn $ "测试集正确率：" ++ show acc

                       putStrLn "\n====【分类别的完整评估指标】===="
                       putStrLn $ "  [(pred, label)]: " ++ show predPairs

                       putStrLn $ "  tp_Lp: " ++ show tp_Lp ++ ", tp_Rp: " ++ show tp_Rp ++ ", tp_Noth: " ++ show tp_Noth
                       putStrLn $ "  fn_Lp: " ++ show fn_Lp ++ ", fn_Rp: " ++ show fn_Rp ++ ", fn_Noth: " ++ show fn_Noth
                       putStrLn $ "  fp_Lp: " ++ show fp_Lp ++ ", fp_Rp: " ++ show fp_Rp ++ ", fp_Noth: " ++ show fp_Noth
                       putStrLn $ "  precision_Lp: " ++ (printf "%.4f" precision_Lp) ++ ", precision_Rp: " ++ (printf "%.4f" precision_Rp) ++ ", precision_Noth: " ++ (printf "%.4f" precision_Noth)
                       putStrLn $ "  recall_Lp: " ++ (printf "%.4f" recall_Lp) ++ ", recall_Rp: " ++ (printf "%.4f" recall_Rp) ++ ", recall_Noth: " ++ (printf "%.4f" recall_Noth)
                       putStrLn $ "  f1Score_Lp: " ++ (printf "%.4f" f1Score_Lp) ++ ", f1Score_Rp: " ++ (printf "%.4f" f1Score_Rp) ++ ", f1Score_Noth: " ++ (printf "%.4f" f1Score_Noth)
                       putStrLn $ "  macro_avg_precision: " ++ (printf "%.4f" macro_avg_precision) ++ ", macro_avg_recall: " ++ (printf "%.4f" macro_avg_recall) ++ ", macro_avg_f1Score: " ++ (printf "%.4f" macro_avg_f1Score)
                       putStrLn $ "  weighted_avg_precision: " ++ (printf "%.4f" weighted_avg_precision) ++ ", weighted_avg_recall: " ++ (printf "%.4f" weighted_avg_recall) ++ ", weighted_avg_f1Score: " ++ (printf "%.4f" weighted_avg_f1Score)
                 else putStrLn "Cancelled."
    else putStrLn "Cancelled."
