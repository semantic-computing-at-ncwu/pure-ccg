-- Copyright (c) 2019-2025 China University of Water Resources and Electric Power,
-- All rights reserved.

module ClusteringSpec where

import Category
import Phrase
import Clustering
import AmbiResol
import Test.Hspec
import qualified Data.Map as Map

spec :: Spec
spec = do
  describe "Clustering" $ do
    it "The result of distPhraSynByIdentity (np, \">\", \"AHn\", 1)) ((s/np)/np, \"<B\", \"AHn\", 2) is 0.75" $ do
      let c1 = getCateFromString "np"
      let c2 = getCateFromString "(s/np)/np"
      let t1 = ">"
      let t2 = "<B"
      let p1 = "AHn"
      let p2 = "AHn"
      let s1 = 1
      let s2 = 2
      distPhraSynByIdentity (c1,t1,p1,s1) (c2,t2,p2,s2) `shouldBe` (0.75 :: Double)

    it "The result of distPhraSyn0ByIdentity (np, \">\", \"AHn\")) ((s/np)/np, \"<B\", \"AHn\") is 2/3" $ do
      let c1 = getCateFromString "np"
      let c2 = getCateFromString "(s/np)/np"
      let t1 = ">"
      let t2 = "<B"
      let p1 = "AHn"
      let p2 = "AHn"
      distPhraSyn0ByIdentity (c1,t1,p1) (c2,t2,p2) `shouldBe` (fromIntegral 2 / fromIntegral 3 :: Double)

-- sim(p1,q1) = 0.5, sim(p1,q2) = 0.25, sim(p2,q1) = 0.0, sim(p2,q2) = 0.25. So sim([p1,p2],[q1,q2]) = (sim(p1,q1) + sim(p2,q2)) / 2.
    it "The result of distPhraSynSetByIdentity [(np, \">\", \"AHn\", 2), (s, \">\", \"DHv\", 3)] [(s/np, \"<B\", \"AHn\", 2) (np, \">B\", \"DHv\", 4)] is 0.625" $ do
      let p1 = (npCate, ">", "AHn", 2)
      let p2 = (sCate, ">", "DHv", 3)
      let q1 = (predCate, "<B", "AHn", 2)
      let q2 = (npCate, ">B", "DHv", 4)
      distPhraSynSetByIdentity [p1,p2] [q1,q2] `shouldBe` (0.625 :: Double)

-- sim(p1,q1) = 2/3, sim(p1,q2) = 1/3, sim(p2,q1) = 1/3, sim(p2,q2) = 1/3. So sim([p1,p2],[q1,q2]) = (sim(p1,q1) + sim(p2,q2)) / 2.
    it "The result of distPhra0SynSetByIdentity [(np, \">\", \"AHn\"), (s, \">\", \"DHv\")] [(s/np, \">\", \"AHn\") (np, \">B\", \"DHv\")] is 2/9" $ do
      let p1 = (npCate, ">", "AHn")
      let p2 = (sCate, ">", "DHv")
      let q1 = (predCate, ">", "AHn")
      let q2 = (npCate, ">B", "DHv")
      distPhraSyn0SetByIdentity [p1,p2] [q1,q2] `shouldBe` (0.5 :: Double)

    it "The result of distPhraSynSetByIdentity sStub nPCs is 0.5 / 3.0" $ do
      let sStub = [(numeralCate, "Desig", "DE", 0), (verbCate, "Desig", "DE", 0), (npCate, "A/n->", "AHn", 2)]
      let nPCs = [(verbCate, "Desig", "DE", 0), (numeralCate, "Desig", "DE", 0), (npCate, ">", "AHn", 1)]
      abs (distPhraSynSetByIdentity sStub nPCs - (0.5 / 3.0 :: Double)) < 1e-5 `shouldBe` True

    it "The result of dist4StruGeneByWeightSum ([(npCate, \">B\", \"HnC\")], (sCate, \">\", \"DHv\"), (npCate,\"<B\",\"AHn\"), [(npCate,\">B\",\"SP\")]), 1, Lp) ([(predCate, \"<\", \"AHn\")], (npCate, \">\", \"AHn\"), (npCate,\"<B\",\"AHn\"), [(predCate,\"<B\",\"AHn\")], 2, Lp) [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] is 0.6666666666666666" $ do
      let sg1 = ([(npCate, ">", "HnC", 2)], (sCate, ">B", "DHv", 3), (npCate,"<B","AHn",2), [(npCate,">B","SP",2)], 1, Lp)
      let sg2 = ([(predCate, "<", "AHn", 3)], (npCate, ">", "AHn", 2), (npCate,"<B","AHn",2), [(predCate,"<B","AHn",3)], 2, Lp)
      let weigthList = [1, 1, 1, 1, 1, 1]
      dist4StruGeneByWeightSum sg1 sg2 weigthList `shouldBe` (4.0 / 6.0 :: Double)

    it "The result of dist4StruGeneByArithMean ([(npCate, \">\", \"HnC\", 2)], (sCate, \">B\", \"DHv\", 3), (npCate,\"<B\",\"AHn\",2), [(npCate,\">B\",\"SP\",2)], 1, Lp) ([(predCate, \"<\", \"AHn\", 3)], (npCate, \">\", \"AHn\", 2), (npCate,\"<B\",\"AHn\", 2), [(predCate,\"<B\",\"AHn\",3)], 2, Lp) is 4.0/6.0" $ do
      let sg1 = ([(npCate,">","HnC",2)], (sCate, ">B", "DHv", 3), (npCate,"<B","AHn",2), [(npCate,">B","SP",2)], 1, Lp)
      let sg2 = ([(predCate,"<","AHn",3)], (npCate, ">", "AHn", 2), (npCate,"<B","AHn",2), [(predCate,"<B","AHn",3)], 2, Lp)
      dist4StruGeneByArithMean sg1 sg2 `shouldBe` (4.0 / 6.0 :: Double)

    it "The result of findDistMinValOfTwoTupleLists [(2, 3.0), (3, 7.5), (6, 9.3), (5, 4.2)] [(3, 4.3), (6, 7.5), (2, 6.2), (5, 4.2)] [] is [(2, 3.0), (3, 4.3), (6, 9.3), (5, 4.2)]" $ do
      let l1 = [(2, 3.0), (3, 7.5), (6, 9.3), (5, 4.2)]
      let l2 = [(3, 4.3), (6, 7.5), (2, 6.2), (5, 4.2)]
      findDistMinValOfTwoTupleLists l1 l2 [] `shouldBe` ([(2, 3.0), (3, 4.3), (6, 7.5), (5, 4.2)] :: SIdxDistList)

    it "The result of readPhraSynListFromStr \"[(np,Desig,DE,1),(np/np,>,XX,2),(s,<,SP,2)]\" is [(np,Desig,DE,1),(np/np,>,XX,2),(s,<,SP,2)] " $ do
      let pSyn1 = readPhraSynFromStr "(np,\"Desig\",\"DE\",1)"
      let pSyn2 = readPhraSynFromStr "(np/np,\">\",\"XX\",2)"
      let pSyn3 = readPhraSynFromStr "(s,\"<\",\"SP\",2)"
      readPhraSynListFromStr "[(np,\"Desig\",\"DE\",1),(np/np,\">\",\"XX\",2),(s,\"<\",\"SP\",2)]" `shouldBe` ([pSyn1, pSyn2, pSyn3] :: [PhraSyn])

    it "The result of getFreqMap [Lp, Lp, Noth, Rp, Noth, Rp, Rp] Map.empty is [(Lp, 2), (Noth,2), (Rp, 3)] " $ do
      let list = [Lp, Lp, Noth, Rp, Noth, Rp, Rp]
      getFreqMap list Map.empty `shouldBe` (Map.fromList [(Lp, 2), (Noth, 2), (Rp, 3)])

    it "The result of getMode4List [Lp, Lp, Noth, Rp, Noth, Rp, Rp] Map.fromList [(Lp, 2), (Noth, 2), (Rp, 3)] is Rp " $ do
      let list = [Lp, Lp, Noth, Rp, Noth, Rp, Rp]
      let kFreqMap = Map.fromList [(Lp, 2), (Noth, 2), (Rp, 3)]
      getMode4List list kFreqMap `shouldBe` (Rp :: Prior)
