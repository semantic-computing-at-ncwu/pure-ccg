{-# LANGUAGE RecordWildCards #-}
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
import Data.Map.Strict as Map hiding (map, foldl, size, filter, splitAt)

import System.Random (randomRIO, mkStdGen, randomR, StdGen, newStdGen)
import List.Shuffle (shuffle_)
import Data.Maybe (fromMaybe)
import Data.Tuple.Utils (fst3, snd3, thd3)
import Control.Monad (when)
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
isLeafNode Empty = True
isLeafNode (Node _ l r) = isLeafNode l && isLeafNode r

-- ===================== 目标动作 One-Hot 转换 =====================
actionToOneHot :: Prior -> Vector Double
actionToOneHot Lp   = LA.fromList [1.0, 0.0, 0.0]
actionToOneHot Rp   = LA.fromList [0.0, 1.0, 0.0]
actionToOneHot Noth = LA.fromList [0.0, 0.0, 1.0]

vecToAction :: Vector Double -> Prior
vecToAction v
  | LA.size v /= 3 = Noth
  | otherwise = case LA.maxIndex v of
      0 -> Lp
      1 -> Rp
      2 -> Noth

-- ===================== 语法树定义 =====================
type NodeFeat = Vector Double
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
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

vecSigmoid :: Vector Double -> Vector Double
vecSigmoid = cmap sigmoid

-- 输入：激活前原始 x
sigmoidDeriv :: Vector Double -> Vector Double
sigmoidDeriv x =
  let v1 = konst 1 (size x) :: Vector Double
      sigVec = cmap sigmoid x
  in sigVec * (v1 - sigVec)

vecTanh :: Vector Double -> Vector Double
vecTanh = cmap tanh

-- 输入：激活前原始 z
tanhDeriv :: Vector Double -> Vector Double
tanhDeriv z = cmap (\x -> 1 - tanh x * tanh x) z

-- 【修复】softmax：减去最大值防止 exp 溢出
softmax :: Vector Double -> Vector Double
softmax vec =
  let m = LA.maxElement vec
      expVec = cmap (\x -> exp (x - m)) vec
      sumExp = sumElements expVec
      sumExpClip = max sumExp 1e-12
  in cmap (/ sumExpClip) expVec

crossEntropy :: Vector Double -> Vector Double -> Double
crossEntropy pred label
  | LA.size pred /= LA.size label = error "crossEntropy: pred/label dimension mismatch"
  | otherwise = - sumElements (LAD.zipVectorWith (\p l -> l * log (max p 1e-8)) pred label)

-- ===================== 网络参数 & 梯度结构 =====================
data NetParam = NetParam
  { leafProjW    :: Matrix Double
  , leafProjB    :: Vector Double
  , nonLeafProjW :: Matrix Double
  , nonLeafProjB :: Vector Double
  , embWeight    :: Matrix Double
  , embBias      :: Vector Double
  , lstmInpW     :: Matrix Double
  , lstmForgetLW :: Matrix Double
  , lstmForgetRW :: Matrix Double
  , lstmOutW     :: Matrix Double
  , lstmCellW    :: Matrix Double
  , lstmInpB     :: Vector Double
  , lstmForgetLB :: Vector Double
  , lstmForgetRB :: Vector Double
  , lstmOutB     :: Vector Double
  , lstmCellB    :: Vector Double
  , clsWeight    :: Matrix Double
  , clsBias      :: Vector Double
  } deriving (Show)

netHidDim :: NetParam -> Int
netHidDim NetParam{embWeight = w} = fst (LA.size w)

netProjDim :: NetParam -> Int
netProjDim netParam = fst (LA.size $ leafProjW netParam)

type NetGrad = NetParam

data LSTMState = LSTMState
                 { hState :: Vector Double
                 , cState :: Vector Double
                 } deriving (Show)

emptyLSTMState :: Int -> LSTMState
emptyLSTMState d = LSTMState (konst 0.0 d) (konst 0.0 d)

-- 前向缓存节点
data NodeCache = NodeCache
  { ncFeat      :: NodeFeat
  , ncProjFeat  :: Vector Double
  , ncEmbOut    :: Vector Double
  , ncInpGatePre :: Vector Double  -- sigmoid 前预激活
  , ncForgetLPre :: Vector Double
  , ncForgetRPre :: Vector Double
  , ncOutGatePre :: Vector Double
  , ncCellCandPre :: Vector Double
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

emptyNodeCache :: Bool -> Int -> Int -> Int -> NodeCache
emptyNodeCache isLeaf featDim projDim hidDim =
  let
      featZero = konst 0.0 featDim
      hidZero = konst 0.0 hidDim
      projZero = konst 0.0 projDim
  in NodeCache
   { ncFeat = featZero
   , ncProjFeat = projZero
   , ncEmbOut = hidZero
   , ncInpGatePre = hidZero
   , ncForgetLPre = hidZero
   , ncForgetRPre = hidZero
   , ncOutGatePre = hidZero
   , ncCellCandPre = hidZero
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
   , ncIsLeaf = isLeaf
   }

data TreeFwdResult = TreeFwdResult
  { tfRootState :: LSTMState
  , tfCacheTree :: SyntaxTree NodeCache
  } deriving (Show)

-- ===================== 特征投影 =====================
projectRawFeat :: NetParam -> Bool -> Vector Double -> Vector Double
projectRawFeat NetParam{..} isLeaf rawFeat =
  let
    expectedDim = if isLeaf
      then snd $ LA.size leafProjW
      else snd $ LA.size nonLeafProjW

    actualDim = size rawFeat
  in
    if actualDim /= expectedDim
      then error $ "projectRawFeat: feature dimension mismatch, expected " ++ show expectedDim ++ ", got " ++ show actualDim
      else if isLeaf
        then leafProjW #> rawFeat + leafProjB
        else nonLeafProjW #> rawFeat + nonLeafProjB

-- ===================== 参数初始化 =====================
randVec :: Int -> StdGen -> (Vector Double, StdGen)
randVec n gen
  | n <= 0    = error "randVec: vector size > 0 required"
  | otherwise =
      let go :: Int -> StdGen -> [Double] -> ([Double], StdGen)
          go 0 g acc = (reverse acc, g)
          go k g acc =
              let (x, g') = randomR (-0.1, 0.1) g
              in go (k-1) g' (x : acc)
          (xs, gen') = go n gen []
          vec = LA.fromList xs
      in (vec, gen')

randMat :: Int -> Int -> StdGen -> (Matrix Double, StdGen)
randMat rows cols gen
  | rows <=0 || cols <=0 = error "randMat: rows/cols >0 required"
  | otherwise =
      let total = rows * cols
          (vec, g') = randVec total gen
          mat = LA.reshape cols vec
      in (mat, g')

initNetParam :: Int -> Int -> Int -> Int -> IO NetParam
initNetParam leafRawDim nonLeafRawDim projDim hidDim = do
  let
    g0 = mkStdGen 42
    (lProjW, g1) = randMat projDim leafRawDim g0
    (lProjB, g2) = randVec projDim g1
    (nlProjW, g3) = randMat projDim nonLeafRawDim g2
    (nlProjB, g4) = randVec projDim g3
    (embW, g5) = randMat hidDim projDim g4
    (embB, g6) = randVec hidDim g5
    (lstmIW, g7) = randMat hidDim hidDim g6
    (lstmFWL, g8) = randMat hidDim hidDim g7
    (lstmFWR, g9) = randMat hidDim hidDim g8
    (lstmOW, g10) = randMat hidDim hidDim g9
    (lstmCW, g11) = randMat hidDim hidDim g10
    (lstmIB, g12) = randVec hidDim g11
    lstmFBL = konst 1.0 hidDim
    lstmFBR = konst 1.0 hidDim
    (lstmOB, g13) = randVec hidDim g12
    (lstmCB, g14) = randVec hidDim g13
    (clsW, g15) = randMat 3 (2 * hidDim) g14
    (clsB, g16) = randVec 3 g15
  return NetParam
    { leafProjW = lProjW, leafProjB = lProjB
    , nonLeafProjW = nlProjW, nonLeafProjB = nlProjB
    , embWeight = embW, embBias = embB
    , lstmInpW = lstmIW, lstmForgetLW = lstmFWL, lstmForgetRW = lstmFWR
    , lstmOutW = lstmOW, lstmCellW = lstmCW
    , lstmInpB = lstmIB, lstmForgetLB = lstmFBL, lstmForgetRB = lstmFBR
    , lstmOutB = lstmOB, lstmCellB = lstmCB
    , clsWeight = clsW, clsBias = clsB
    }

zeroGrad :: Int -> Int -> Int -> Int -> NetGrad
zeroGrad leafRawDim nonLeafRawDim projDim hidDim =
  let zm r c = konst 0.0 (r, c)
      zv d = konst 0.0 d
  in NetParam
    { leafProjW    = zm projDim leafRawDim
    , leafProjB    = zv projDim
    , nonLeafProjW = zm projDim nonLeafRawDim
    , nonLeafProjB = zv projDim
    , embWeight = zm hidDim projDim, embBias = zv hidDim
    , lstmInpW  = zm hidDim hidDim, lstmForgetLW = zm hidDim hidDim, lstmForgetRW = zm hidDim hidDim
    , lstmOutW  = zm hidDim hidDim, lstmCellW = zm hidDim hidDim
    , lstmInpB  = zv hidDim, lstmForgetLB = zv hidDim, lstmForgetRB = zv hidDim
    , lstmOutB  = zv hidDim, lstmCellB = zv hidDim
    , clsWeight = zm 3 (2 * hidDim), clsBias = zv 3
    }

gradAdd :: NetGrad -> NetGrad -> NetGrad
gradAdd g1 g2 = NetParam
    { leafProjW    = leafProjW g1 + leafProjW g2
    , leafProjB    = leafProjB g1 + leafProjB g2
    , nonLeafProjW = nonLeafProjW g1 + nonLeafProjW g2
    , nonLeafProjB = nonLeafProjB g1 + nonLeafProjB g2
    , embWeight = embWeight g1 + embWeight g2
    , embBias   = embBias g1 + embBias g2
    , lstmInpW  = lstmInpW g1 + lstmInpW g2
    , lstmForgetLW = lstmForgetLW g1 + lstmForgetLW g2
    , lstmForgetRW = lstmForgetRW g1 + lstmForgetRW g2
    , lstmOutW  = lstmOutW g1 + lstmOutW g2
    , lstmCellW = lstmCellW g1 + lstmCellW g2
    , lstmInpB  = lstmInpB g1 + lstmInpB g2
    , lstmForgetLB = lstmForgetLB g1 + lstmForgetLB g2
    , lstmForgetRB = lstmForgetRB g1 + lstmForgetRB g2
    , lstmOutB  = lstmOutB g1 + lstmOutB g2
    , lstmCellB = lstmCellB g1 + lstmCellB g2
    , clsWeight = clsWeight g1 + clsWeight g2
    , clsBias   = clsBias g1 + clsBias g2
    }

gradScale :: Double -> NetGrad -> NetGrad
gradScale s g1 =
   NetParam
    { leafProjW    = scale s (leafProjW g1)
    , leafProjB    = scale s (leafProjB g1)
    , nonLeafProjW = scale s (nonLeafProjW g1)
    , nonLeafProjB = scale s (nonLeafProjB g1)
    , embWeight = scale s (embWeight g1)
    , embBias   = scale s (embBias g1)
    , lstmInpW  = scale s (lstmInpW g1)
    , lstmForgetLW = scale s (lstmForgetLW g1)
    , lstmForgetRW = scale s (lstmForgetRW g1)
    , lstmOutW  = scale s (lstmOutW g1)
    , lstmCellW = scale s (lstmCellW g1)
    , lstmInpB  = scale s (lstmInpB g1)
    , lstmForgetLB = scale s (lstmForgetLB g1)
    , lstmForgetRB = scale s (lstmForgetRB g1)
    , lstmOutB  = scale s (lstmOutB g1)
    , lstmCellB = scale s (lstmCellB g1)
    , clsWeight = scale s (clsWeight g1)
    , clsBias   = scale s (clsBias g1)
    }

-- 梯度裁剪（防止爆炸）
clipGrad :: Double -> NetGrad -> NetGrad
clipGrad maxNorm g =
  let vecNorm v = sqrt $ sumElements (v * v)
      matNorm m = sqrt $ sumElements (m * m)
      collectNorms = [ matNorm (leafProjW g), vecNorm (leafProjB g)
                     , matNorm (nonLeafProjW g), vecNorm (nonLeafProjB g)
                     , matNorm (embWeight g), vecNorm (embBias g)
                     , matNorm (lstmInpW g), matNorm (lstmForgetLW g), matNorm (lstmForgetRW g)
                     , matNorm (lstmOutW g), matNorm (lstmCellW g)
                     , vecNorm (lstmInpB g), vecNorm (lstmForgetLB g), vecNorm (lstmForgetRB g)
                     , vecNorm (lstmOutB g), vecNorm (lstmCellB g)
                     , matNorm (clsWeight g), vecNorm (clsBias g)
                     ]
      totalNorm = sqrt $ sum (map (^2) collectNorms)
      scaleFactor = if totalNorm > maxNorm then maxNorm / totalNorm else 1.0
  in gradScale scaleFactor g

updateParam :: Double -> NetGrad -> NetParam -> NetParam
updateParam lr grad p =
  let
    updateMat m gm = m + scale (-lr) gm
    updateVec v gv = v + scale (-lr) gv
  in p
    { leafProjW    = updateMat (leafProjW p) (leafProjW grad)
    , leafProjB    = updateVec (leafProjB p) (leafProjB grad)
    , nonLeafProjW = updateMat (nonLeafProjW p) (nonLeafProjW grad)
    , nonLeafProjB = updateVec (nonLeafProjB p) (nonLeafProjB grad)
    , embWeight = updateMat (embWeight p) (embWeight grad)
    , embBias   = updateVec (embBias p) (embBias grad)
    , lstmInpW  = updateMat (lstmInpW p) (lstmInpW grad)
    , lstmForgetLW = updateMat (lstmForgetLW p) (lstmForgetLW grad)
    , lstmForgetRW = updateMat (lstmForgetRW p) (lstmForgetRW grad)
    , lstmOutW  = updateMat (lstmOutW p) (lstmOutW grad)
    , lstmCellW = updateMat (lstmCellW p) (lstmCellW grad)
    , lstmInpB  = updateVec (lstmInpB p) (lstmInpB grad)
    , lstmForgetLB = updateVec (lstmForgetLB p) (lstmForgetLB grad)
    , lstmForgetRB = updateVec (lstmForgetRB p) (lstmForgetRB grad)
    , lstmOutB  = updateVec (lstmOutB p) (lstmOutB grad)
    , lstmCellB = updateVec (lstmCellB p) (lstmCellB grad)
    , clsWeight = updateMat (clsWeight p) (clsWeight grad)
    , clsBias   = updateVec (clsBias p) (clsBias grad)
    }

-- ===================== 【修复】标准 Child-Sum Tree-LSTM 前向传播 =====================
treeForwardCached :: NetParam -> RawSyntaxTree -> TreeFwdResult
treeForwardCached param EmptySTree =             -- 空节点不参与参数更新，缓存仅占位
  let
    hidDim = netHidDim param
  in TreeFwdResult (emptyLSTMState hidDim) EmptySTree

treeForwardCached param st@STreeNode{nodeData = feat, stLeft = l, stRight = r, stIsLeaf = isLeafNode} =
  let
    leftRes = treeForwardCached param l
    rightRes = treeForwardCached param r
    LSTMState hl cl = tfRootState leftRes
    LSTMState hr cr = tfRootState rightRes
    NetParam{..} = param

    hidDim = netHidDim param
    projFeat = projectRawFeat param isLeafNode feat
    embOut = embWeight #> projFeat + embBias

    -- ========== 修复：输入门/输出门/cell候选 并入左右子树隐状态 ==========
    inpGatePre     = lstmInpW #> embOut + lstmForgetLW #> hl + lstmForgetRW #> hr + lstmInpB
    forgetLPre     = lstmForgetLW #> hl + lstmForgetLB
    forgetRPre     = lstmForgetRW #> hr + lstmForgetRB
    outGatePre     = lstmOutW #> embOut + lstmForgetLW #> hl + lstmForgetRW #> hr + lstmOutB
    cellCandPre    = lstmCellW #> embOut + lstmForgetLW #> hl + lstmForgetRW #> hr + lstmCellB

    inpGate     = vecSigmoid inpGatePre
    forgetLGate = vecSigmoid forgetLPre
    forgetRGate = vecSigmoid forgetRPre
    outGate     = vecSigmoid outGatePre
    cellCand    = vecTanh cellCandPre

    newC = forgetLGate * cl + forgetRGate * cr + inpGate * cellCand
    newH = outGate * vecTanh newC

    cache = NodeCache
      { ncFeat = feat
      , ncProjFeat = projFeat
      , ncEmbOut = embOut
      , ncInpGatePre = inpGatePre
      , ncForgetLPre = forgetLPre
      , ncForgetRPre = forgetRPre
      , ncOutGatePre = outGatePre
      , ncCellCandPre = cellCandPre
      , ncInpGate = inpGate, ncForgetLG = forgetLGate, ncForgetRG = forgetRGate
      , ncOutGate = outGate, ncCellCand = cellCand, ncNewC = newC, ncNewH = newH
      , ncLeftH = hl, ncLeftC = cl, ncRightH = hr, ncRightC = cr
      , ncIsLeaf = isLeafNode
      }
    cachedNode = STreeNode cache (tfCacheTree leftRes) (tfCacheTree rightRes) isLeafNode
  in TreeFwdResult (LSTMState newH newC) cachedNode

-- ===================== 反向传播 Node BP =====================
backwardNode :: NetParam
             -> NodeCache
             -> Vector Double
             -> Vector Double
             -> (NetGrad, Vector Double, Vector Double, Vector Double, Vector Double, Vector Double)
backwardNode NetParam{..} nc@NodeCache{..} dLdh dLdc =
  let
    tanhNewC = vecTanh ncNewC
    dTanhC = tanhDeriv ncNewC
    dh_dc = ncOutGate * dTanhC
    dLdnewC = dLdh * dh_dc + dLdc

    dLdOutGate = dLdh * tanhNewC
    dRawOut = dLdOutGate * sigmoidDeriv ncOutGatePre

    dLdInpGate = dLdnewC * ncCellCand
    dRawInp = dLdInpGate * sigmoidDeriv ncInpGatePre

    dLdForgetL = dLdnewC * ncLeftC
    dRawForgetL = dLdForgetL * sigmoidDeriv ncForgetLPre

    dLdForgetR = dLdnewC * ncRightC
    dRawForgetR = dLdForgetR * sigmoidDeriv ncForgetRPre

    dLdCellCand = dLdnewC * ncInpGate
    dRawCell = dLdCellCand * tanhDeriv ncCellCandPre

    dEmbLSTM = tr' lstmInpW #> dRawInp
             + tr' lstmOutW #> dRawOut
             + tr' lstmCellW #> dRawCell
             + tr' lstmForgetLW #> dRawForgetL
             + tr' lstmForgetRW #> dRawForgetR

    dProjFeat = tr' embWeight #> dEmbLSTM

    dLdhL = tr' lstmForgetLW #> dRawForgetL
    dLdcL = dLdnewC * ncForgetLG
    dLdhR = tr' lstmForgetRW #> dRawForgetR
    dLdcR = dLdnewC * ncForgetRG

    hidDim = LA.size ncEmbOut
    projDim = LA.size dProjFeat
    leafRawDim = snd $ LA.size leafProjW
    nonLeafRawDim = snd $ LA.size nonLeafProjW
    zeroG = zeroGrad leafRawDim nonLeafRawDim projDim hidDim

    (projGrad, dRawFeat)
      | ncIsLeaf =
          let gW = outer dProjFeat ncFeat
              gB = dProjFeat
              dX = tr' leafProjW #> dProjFeat
              pg = zeroG { leafProjW = gW, leafProjB = gB }
          in (pg, dX)
      | otherwise =
          let gW = outer dProjFeat ncFeat
              gB = dProjFeat
              dX = tr' nonLeafProjW #> dProjFeat
              pg = zeroG { nonLeafProjW = gW, nonLeafProjB = gB }
          in (pg, dX)

    gradEmbW = outer dEmbLSTM ncProjFeat
    gradEmbB = dEmbLSTM
    gradIW = outer dRawInp ncEmbOut
    gradIB = dRawInp
    gradFWL = outer dRawForgetL ncLeftH
    gradFBL = dRawForgetL
    gradFWR = outer dRawForgetR ncRightH
    gradFBR = dRawForgetR
    gradOW = outer dRawOut ncEmbOut
    gradOB = dRawOut
    gradCW = outer dRawCell ncEmbOut
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

-- ===================== 整树反向传播【修复空树梯度】 =====================
backwardTree :: NetParam -> SyntaxTree NodeCache -> Vector Double -> Vector Double -> NetGrad
backwardTree param EmptySTree _ _ =
  let
    leafRawDim = snd $ LA.size (leafProjW param)
    nonLeafRawDim = snd $ LA.size (nonLeafProjW param)
    projDim = fst $ LA.size (leafProjW param)
    hidDim = fst $ LA.size (embWeight param)
  in zeroGrad leafRawDim nonLeafRawDim projDim hidDim
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
    dLogit = pred - label
    gradClsW = outer dLogit concatH
    gradClsB = dLogit
    hidDim = LA.size hl
    dConcatH = tr' clsWeight #> dLogit
    dHl = subVector 0 hidDim dConcatH
    dHr = subVector hidDim (LA.size dConcatH - hidDim) dConcatH
    gradL = backwardTree param (tfCacheTree lFwd) dHl (konst 0.0 hidDim)
    gradR = backwardTree param (tfCacheTree rFwd) dHr (konst 0.0 hidDim)
    totalGrad0 = gradL `gradAdd` gradR
    totalGrad = totalGrad0
      { clsWeight = gradClsW
      , clsBias = gradClsB
      }
  in (loss, totalGrad)

trainStep :: NetParam -> TrainSample -> Double -> (Double, NetParam)
trainStep param sample lr =
  let (loss, grad) = sampleForwardLossGrad param sample
      clippedGrad = clipGrad 5.0 grad
      newParam = updateParam lr clippedGrad param
  in (loss, newParam)

batchTrainStep :: NetParam -> [TrainSample] -> Double -> (Double, NetParam)
batchTrainStep param samples lr =
  let
    leafD = snd $ LA.size $ leafProjW param
    nonLeafD = snd $ LA.size $ nonLeafProjW param
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
    clippedGrad = clipGrad 5.0 avgGrad
    newParam = updateParam lr clippedGrad param
  in (avgLoss, newParam)

-- ===================== 树构造辅助函数（测试用） =====================
mkLeaf :: [Double] -> [Double] -> RawSyntaxTree
mkLeaf wordEmb catEmb =
  let feat = LA.fromList (wordEmb ++ catEmb)
  in STreeNode feat EmptySTree EmptySTree True

mkNonLeaf :: [Double] -> [Double] -> [Double] -> RawSyntaxTree -> RawSyntaxTree -> RawSyntaxTree
mkNonLeaf cat tag phra l r =
  let feat = LA.fromList (cat ++ tag ++ phra)
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

-- ===================== 测试入口：可独立运行 =====================
testPipeline :: IO ()
testPipeline = do
  putStrLn "==== Binary Tree-LSTM 【修复完整版】 ===="
  let
    dim_word    = 2
    dim_cat     = 2
    dim_tag     = 2
    dim_stru    = 2
    leafRawDim    = dim_word + dim_cat
    nonLeafRawDim = dim_cat + dim_tag + dim_stru
    projDim       = 4
    hidDim        = 8
    lr      = 0.008
    epochs  = 800
  initP <- initNetParam leafRawDim nonLeafRawDim projDim hidDim
  let
    leafA = mkLeaf [0.1,0.3] [0.5,0.2]
    leafB = mkLeaf [0.4,0.2] [0.1,0.7]
    treeL = mkNonLeaf [0.3,0.6] [0.2,0.4] [0.1,0.9] leafA EmptySTree
    treeR = mkNonLeaf [0.7,0.2] [0.5,0.1] [0.3,0.4] EmptySTree leafB
    testSampleSet = [ ((treeL, treeR), Rp) ]
  samples <- pure testSampleSet
  putStrLn $ "样本数量：" ++ show (length samples)
  trainedP <- trainLoop initP samples lr epochs
  let ((testL,testR), trueAct) = head samples
      lFwd = treeForwardCached trainedP testL
      rFwd = treeForwardCached trainedP testR
      LSTMState hl _ = tfRootState lFwd
      LSTMState hr _ = tfRootState rFwd
      concatH = LA.vjoin [hl,hr]
      logits = clsWeight trainedP #> concatH + clsBias trainedP
      pred = softmax logits
      predAct = vecToAction pred
  putStrLn "\n====预测结果===="
  putStrLn $ "预测分布:" ++ show pred
  putStrLn $ "预测动作:" ++ show predAct
  putStrLn $ "真实动作:" ++ show trueAct

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
 -}
biTreePhraSyn0Seman2SyntaxTreeNodeFeat :: BiTree (PhraSyn0, Term)
                                     -> Map Category (Vector Double)
                                     -> Map Tag (Vector Double)
                                     -> Map PhraStru (Vector Double)
                                     -> Int
                                     -> Int
                                     -> Int
                                     -> IO (SyntaxTree NodeFeat)
biTreePhraSyn0Seman2SyntaxTreeNodeFeat Empty _ _ _ _ _ _ = return EmptySTree
biTreePhraSyn0Seman2SyntaxTreeNodeFeat (Node (phraSyn0, semanTerm) lt rt) cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru = do
    let
        (v1, g1) = randVec dim_cate (mkStdGen 42)
        (v2, g2) = randVec dim_tag g1
        (v3, _) = randVec dim_stru g2
        cateVec = fromMaybe v1 $ Map.lookup (fst3 phraSyn0) cateVecMap
        tagVec = fromMaybe v2 $ Map.lookup (snd3 phraSyn0) tagVecMap
        struVec = fromMaybe v3 $ Map.lookup (thd3 phraSyn0) struVecMap
{-
    putStrLn $ "  seman: " ++ show semanTerm
    putStrLn $ "  cateVec: " ++ show cateVec
    putStrLn $ "  tagVec: " ++ show tagVec
    putStrLn $ "  struVec: " ++ show struVec
 -}
    confInfo <- readFile "Configuration"
    let wordEmbTbl = getConfProperty "wordEmbTbl" confInfo
    conn <- getConn
    let sqlstat = Query $ DS.fromString ("select CAST(emb AS CHAR) from " ++ wordEmbTbl ++ " where word = ?")

    if isPrimTerm semanTerm
      then do                 -- Word seman
--        putStrLn $ "  It's a primitive term."
        let seman = init $ show semanTerm
        (defs, is) <- query conn sqlstat [MySQLText (T.pack seman)]       -- Search database table 'wordEmbTbl'
        rows <- S.toList is
--        putStrLn $ "  Size of rows: " ++ show (length rows)
        if rows == []                      -- OOV
          then do
            (defs, is) <- query conn sqlstat [toMySQLText "<OOV>"]
            rows' <- S.toList is
            if rows' == []                 -- [[MySQLValue]]
              then error "biTreePhraSyn0Seman2SyntaxTreeNodeFeat: OOV is NOT in 'wordEmbTbl'."
              else do                      -- rows' = [[MySQLBytes]]
                let embVec = (fromMaybe' . fromMySQLJSON) ((rows'!!0)!!0)     -- decode :: ByteString -> Maybe [Double]
                putStrLn $ seman ++ "  Use <OOV> vector"
                return STreeNode { nodeData = LA.vjoin [cateVec, LA.fromList embVec]
                                 , stLeft = EmptySTree
                                 , stRight = EmptySTree
                                 , stIsLeaf = True
                                 }
          else do                         -- Not OOV
            let embVec = (fromMaybe' . fromMySQLJSON) ((rows!!0)!!0)     -- [Double]
--            putStrLn $ "  embVec(" ++ seman ++ "): " ++ show embVec
            return STreeNode { nodeData = LA.vjoin [cateVec, LA.fromList embVec]
                             , stLeft = EmptySTree
                             , stRight = EmptySTree
                             , stIsLeaf = True
                             }
      else do           -- Phrasal seman
--        putStrLn $ "  It's a compound term."
        lt <- biTreePhraSyn0Seman2SyntaxTreeNodeFeat lt cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru
        rt <- biTreePhraSyn0Seman2SyntaxTreeNodeFeat rt cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru
        return STreeNode { nodeData = LA.vjoin [cateVec, tagVec, struVec]
                         , stLeft = lt
                         , stRight = rt
                         , stIsLeaf = False
                         }

buildSampleSet :: Int -> Int -> Int -> IO [SampleRecord]
buildSampleSet dim_cate dim_tag dim_stru = do
  confInfo <- readFile "Configuration"
  let syntax_ambig_resol_model = getConfProperty "syntax_ambig_resol_model" confInfo
      syntax_resol_sample_startId = getConfProperty "syntax_resol_sample_startId" confInfo
      syntax_resol_sample_endId = getConfProperty "syntax_resol_sample_endId" confInfo
      startId = read syntax_resol_sample_startId :: Int
      endId = read syntax_resol_sample_endId :: Int

  putStrLn $ " syntax_ambig_resol_model: " ++ syntax_ambig_resol_model
  putStrLn $ " syntax_resol_sample_startId: " ++ syntax_resol_sample_startId
  putStrLn $ " syntax_resol_sample_endId: " ++ syntax_resol_sample_endId

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

  putStrLn $ "lotLosRotRosPriorList: " ++ show lotLosRotRosPriorList
  let loRoPriorList = map (\x -> (( mergeBiTree (fst5 x) (snd5 x)                -- BiTree (PhraSyn0, Term)
                                  , mergeBiTree (thd5 x) (fth5 x)
                                  )
                                  , fif5 x
                                 )
                          ) lotLosRotRosPriorList           -- [((BiTree (PhraSyn0, Term), BiTree (PhraSyn0, Term)), Prior)]
  putStrLn $ "Sample number = " ++ show (length lotLosRotRosPriorList) ++ ", the first is: " ++ show (loRoPriorList!!0)

  cateVecMap <- cate2Vec                         -- Map Category (Vector Double)
  tagVecMap <- tag2Vec                           -- Map Tag (Vector Double)
  struVecMap <- stru2Vec                         -- Map PhraStru (Vector Double)

  sampleSet <- mapM (\x -> do
                            lt <- biTreePhraSyn0Seman2SyntaxTreeNodeFeat ((fst . fst) x) cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru
                            rt <- biTreePhraSyn0Seman2SyntaxTreeNodeFeat ((snd . fst) x) cateVecMap tagVecMap struVecMap dim_cate dim_tag dim_stru
                            let prior = snd x
                            return SampleRecord{ recLeftTree = lt
                                               , recRightTree = rt
                                               , recTargetAct = prior
                                               }
                     ) loRoPriorList
  putStrLn $ "buildSampleSet: sampleSet!!0: " ++ show (sampleSet!!0)
  return sampleSet

-- 评估整个数据集上模型预测的平均损失率
evalLoss :: NetParam -> [TrainSample] -> Double
evalLoss param samples = avgLoss
  where
    leafD = snd $ LA.size $ leafProjW param
    nonLeafD = snd $ LA.size $ nonLeafProjW param
    projD = netProjDim param
    hidD = netHidDim param
    initG = zeroGrad leafD nonLeafD projD hidD
    pairs = map (sampleForwardLossGrad param) samples
    losses = map fst pairs
    totalLoss = sum losses
    sampleSetSize = fromIntegral (length samples)
    avgLoss = totalLoss / sampleSetSize

-- ========== 【trainLoop2：支持验证 + 早停 + 保存最优参数】 ==========
-- 入参：初始参数、训练集、验证集、学习率、最大轮数、早停耐心值
-- 返回：验证集上最优模型参数
trainLoop2 :: NetParam -> [TrainSample] -> [TrainSample] -> Double -> Int -> Int -> IO NetParam
trainLoop2 initParam trainSet valSet lr maxEpoch patience = go initParam maxEpoch patience 1 1e9
  where
    go :: NetParam -> Int -> Int -> Int -> Double -> IO NetParam
    go bestParam remainEpoch remainPatience epoch bestValLoss
      | remainEpoch <= 0 = do
          putStrLn $ "达到最大 Epoch(" ++ show maxEpoch ++ ")，停止训练，返回最优模型"
          return bestParam
      | remainPatience <= 0 = do
          putStrLn $ "早停触发！连续 " ++ show patience ++ " 轮验证损失无下降，终止训练"
          return bestParam
      | otherwise = do
          putStrLn $ "\n==== Epoch " ++ show epoch ++ " ===="
          -- 1. 完整一轮训练：遍历训练集更新参数
          currParam <- trainOneEpoch initParam trainSet lr
          -- 2. 在验证集评估
          let valLoss = evalLoss currParam valSet
          putStrLn $ "Epoch " ++ show epoch ++ " | 验证集平均损失: " ++ show valLoss
          -- 3. 判断是否更新最优模型
          if valLoss < bestValLoss
            then do
              putStrLn $ "验证损失下降，更新最优模型，新最优损失: " ++ show valLoss
              go currParam (remainEpoch - 1) patience (epoch + 1) valLoss
            else do
              putStrLn $ "验证损失未下降，剩余耐心: " ++ show (remainPatience - 1)
              go bestParam (remainEpoch - 1) (remainPatience - 1) (epoch + 1) bestValLoss

-- 单轮训练：遍历trainSet，反向传播更新权重
trainOneEpoch :: NetParam -> [TrainSample] -> Double -> IO NetParam
trainOneEpoch param samples lr = do
  gen <- mkStdGen <$> randomRIO (1,99999)
  let shuffled = shuffle_ samples gen
      (_, newParam) = batchTrainStep param shuffled lr
  return newParam

--- ===================== 训练与测试：可独立运行 =====================
testPipeline2 :: IO ()
testPipeline2 = do
  putStrLn "==== Binary Tree-LSTM ===="
  confInfo <- readFile "Configuration"
  let
    syntax_ambig_resol_model = getConfProperty "syntax_ambig_resol_model" confInfo
    syntax_resol_sample_startId_str = getConfProperty "syntax_resol_sample_startId" confInfo
    syntax_resol_sample_endId_str = getConfProperty "syntax_resol_sample_endId" confInfo
    dim_word_str = getConfProperty "dim_word" confInfo
    dim_cate_str = getConfProperty "dim_cate" confInfo
    dim_tag_str = getConfProperty "dim_tag" confInfo
    dim_stru_str = getConfProperty "dim_stru" confInfo
    dim_proj_str = getConfProperty "dim_proj" confInfo
    dim_hidden_str = getConfProperty "dim_hidden" confInfo
    lr_str = getConfProperty "lr" confInfo
    epochs_str = getConfProperty "epochs" confInfo
    earlyStopPatience_str = getConfProperty "earlyStopPatience" confInfo

    dim_word = read dim_word_str :: Int
    dim_cate = read dim_cate_str :: Int
    dim_tag = read dim_tag_str :: Int
    dim_stru = read dim_stru_str :: Int
    projDim = read dim_proj_str :: Int
    hidDim = read dim_hidden_str :: Int
    lr = read lr_str :: Double
    epochs = read epochs_str :: Int
    earlyStopPatience = read earlyStopPatience_str :: Int

    leafRawDim    = dim_word + dim_cate
    nonLeafRawDim = dim_cate + dim_tag + dim_stru

  putStrLn $ " syntax_ambig_resol_model: " ++ syntax_ambig_resol_model
  putStrLn $ " syntax_resol_sample_startId: " ++ syntax_resol_sample_startId_str
  putStrLn $ " syntax_resol_sample_endId: " ++ syntax_resol_sample_endId_str
  putStrLn $ " dim_word: " ++ dim_word_str
  putStrLn $ " dim_cate: " ++ dim_cate_str
  putStrLn $ " dim_tag: " ++ dim_tag_str
  putStrLn $ " dim_stru: " ++ dim_stru_str
  putStrLn $ " projDim: " ++ dim_proj_str
  putStrLn $ " hidDim: " ++ dim_hidden_str
  putStrLn $ " lr: " ++ lr_str
  putStrLn $ " epochs: " ++ epochs_str
  putStrLn $ " earlyStopPatience: " ++ earlyStopPatience_str

  let prompt = " Test the building of sample set in Module BiTreeLSTM, are you sure? [y/n] (RETURN for 'y'): "
  answer <- getLineUntil prompt ["y","n"] True
  if answer == "y"
    then do
      initP <- initNetParam leafRawDim nonLeafRawDim projDim hidDim
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
      bestTrainedP <- trainLoop2 initP trainSet valSet lr epochs earlyStopPatience

      -- ========== 【最终评估：独立测试集，仅执行一次】 ==========
      putStrLn "\n====【最终测试集评估】===="
      case testSet of
        [] -> putStrLn "警告：测试集为空，无法评估！"
        (firstSample:_) -> do
          -- 1. 取第一条做可视化展示（原有逻辑保留）
          let ((testL,testR), trueAct) = firstSample
              lFwd = treeForwardCached bestTrainedP testL
              rFwd = treeForwardCached bestTrainedP testR
              LSTMState hl _ = tfRootState lFwd
              LSTMState hr _ = tfRootState rFwd
              concatH = LA.vjoin [hl,hr]
              logits = clsWeight bestTrainedP #> concatH + clsBias bestTrainedP
              pred = softmax logits
              predAct = vecToAction pred
          putStrLn "\n====【测试集第一条样例预测展示】===="
          putStrLn $ "预测分布:" ++ show pred
          putStrLn $ "预测动作:" ++ show predAct
          putStrLn $ "真实动作:" ++ show trueAct

          -- 2. 遍历全部测试集，计算整体指标（正式评估）
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
          putStrLn "\n====【完整测试集正式评估结果】===="
          putStrLn $ "测试集平均损失：" ++ show totalLoss
          putStrLn $ "正确样本数：" ++ show correct ++ " / " ++ show total
          putStrLn $ "测试集准确率：" ++ show acc

    else putStrLn "Cancelled."
