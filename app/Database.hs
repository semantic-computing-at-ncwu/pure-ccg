{-# LANGUAGE OverloadedStrings #-}

-- Copyright (c) 2019-2026 North China University of Water Resources and Electric Power
-- All rights reserved.

module Database (
  fromMySQLInt8,               -- MySQLValue (MySQLInt8) -> Int
  fromMySQLInt16U,             -- MySQLValue (MySQLInt16U) -> Int
  fromMySQLInt32U,             -- MySQLValue (MySQLInt32U) -> Int
  fromMySQLInt32,              -- MySQLValue (MySQLInt32) -> Int
  fromMySQLInt64U,             -- MySQLValue (MySQLInt64U) -> Int
  fromMySQLInt64,              -- MySQLValue (MySQLInt64) -> Int
  fromMySQLDecimal,            -- MySQLValue (MySQLDecimal) -> Int
  fromMySQLText,               -- MySQLValue (MySQLText) -> String
  fromMySQLNullText,           -- MySQLValue (MySQLNull)-> String
  fromMySQLNullVarchar,        -- MySQLValue (MySQLNull) -> String
  fromMySQLFloat,              -- MySQLValue (MySQLFloat) -> Float
  fromMySQLDouble,             -- MySQLValue (MySQLDouble) -> Double
  toMySQLInt8,                 -- Int -> MySQLValue (MySQLInt8)
  toMySQLInt16U,               -- Int -> MySQLValue (MySQLInt16U)
  toMySQLInt32U,               -- Int -> MySQLValue (MySQLInt32U)
  toMySQLInt32,                -- Int -> MySQLValue (MySQLInt32)
  toMySQLInt64U,               -- Int -> MySQLValue (MySQLInt64U)
  toMySQLInt64,                -- Int -> MySQLValue (MySQLInt64)
  toMySQLDecimal,              -- Int -> MySQLValue (MySQLDecimal)
  toMySQLBytes,                -- String -> MySQLValue (MySQLBytes)
  toMySQLText,                 -- String -> MySQLValue (MySQLText)
  toMySQLNullText,             -- MySQLValue (MySQLNull)
  toMySQLNullVarchar,          -- MySQLValue (MySQLNull)
  toMySQLFloat,                -- Float -> MySQLValue (MySQLFloat)
  toMySQLDouble,               -- Double -> MySQLValue (MySQLDouble)
  getColumnDB,                 -- ColumnDef -> ByteString
  getColumnTable,              -- ColumnDef -> ByteString
  getColumnOrigTable,          -- ColumnDef -> ByteString
  getColumnName,               -- ColumnDef -> ByteString
  getColumnOrigName,           -- ColumnDef -> ByteString
  getColumnCharSet,            -- ColumnDef -> Word16
  getColumnLength,             -- ColumnDef -> Word32
  getColumnType,               -- ColumnDef -> String
  getColumnFlags,              -- ColumnDef -> Word16
  getColumnDecimals,           -- ColumnDef -> Word8
  getOkAffectedRows,           -- OK -> Int
  getOkLastInsertID,           -- OK -> Int
  getOkStatus,                 -- OK -> Word16
  getOkWarningCnt,             -- OK -> Word16
  readStreamByText,            -- [String] -> S.InputStream [MySQLValue] -> IO [String]
  readStreamByInt64,           -- [Int] -> S.InputStream [MySQLValue] -> IO [Int]
  readStreamByInt8Decimal,         -- [(Int,Double)] -> S.InputStream [MySQLValue] -> IO [(Int,Double)]
  readStreamByInt8Int64,           -- [[Int]] -> S.InputStream [MySQLValue] -> IO [[Int]]
  readStreamByInt32Text,           -- [(Int, String)] -> S.InputStream [MySQLValue] -> IO [(Int, String)]
  readStreamByInt32UText,          -- [(Int, String)] -> S.InputStream [MySQLValue] -> IO [(Int, String)]
  readStreamByInt32TextText,       -- [(Int, String, String)] -> S.InputStream [MySQLValue] -> IO [(Int, String, String)]
  readStreamByInt32UTextTextText,          -- [(Int, String, String, String)] -> S.InputStream [MySQLValue] -> IO [(Int, String, String, String)]
  readStreamByTextText,            -- [(String, String)] -> S.InputStream [MySQLValue] -> IO [(String, String)]
  readStreamByTextTextTextText,    -- [(String, String, String, String)] -> S.InputStream [MySQLValue] -> IO [(String, String, String, String)]
  readStreamByInt32UTextText,      -- [(Int, (String, String))] -> S.InputStream [MySQLValue] -> IO [(Int, (String, String))]
  readStreamByInt32UTextTextTextTextInt8,    -- [(Int, (String, String, String, String, Int))] -> S.InputStream [MySQLValue] -> IO [(Int, (String, String, String, String, Int))]
  readStreamByTextTextInt8,        -- [String] -> S.InputStream [MySQLValue] -> IO [String]
  readStreamByTextTextInt8Text,    -- [(String, String, Double)] -> S.InputStream [MySQLValue] -> IO [(String, String, Double)]
  readStreamByTextTextDouble,      -- [String] -> S.InputStream [MySQLValue] -> IO [String]
  getConn,                     -- IO MySQLConn
  getConnByUserWqj,            -- IO MySQLConn
  recogOk,                     -- IO ()
  wordEmbedFile2MySQL,         -- IO ()
  ) where

import           Control.Monad
import           System.IO (getLine, openFile, hClose, hIsEOF, Handle, IOMode(..))
import qualified System.IO as SIO
import qualified System.IO.Streams as S
import qualified Data.String as DS
import           Database.MySQL.Base
import           Data.Scientific
import           Data.List.Utils
import           Data.List as DL
import           Data.Tuple.Utils
import           Data.Int
import           Data.Text as DT hiding (length, map, head, tail, last, foldl)
import           Data.Word
import           Data.ByteString.Char8 as BC hiding (putStr, putStrLn, readFile, map, head, tail, last, length)
import           Utils
import           Phrase(getPhraCateFromString, getPhraCateListFromString)

import qualified Data.Text.IO as TIO
import           Data.Aeson (encode)
import qualified Data.ByteString.Lazy as BL
import           Data.ByteString (toStrict)
import           Data.Text.Encoding (decodeUtf8)

fromMySQLInt8 :: MySQLValue -> Int
fromMySQLInt8 (MySQLInt8 a) = read (show a) :: Int
fromMySQLInt8 _ = error "fromMySQLInt8: Parameter error."

fromMySQLInt16U :: MySQLValue -> Int
fromMySQLInt16U (MySQLInt16U a) = read (show a) :: Int
fromMySQLInt16U _ = error "fromMySQLInt16U: Parameter error."

fromMySQLInt32U :: MySQLValue -> Int
fromMySQLInt32U (MySQLInt32U a) = read (show a) :: Int
fromMySQLInt32U _ = error "fromMySQLInt32U: Parameter error."

fromMySQLInt32 :: MySQLValue -> Int
fromMySQLInt32 (MySQLInt32 a) = read (show a) :: Int
fromMySQLInt32 _ = error "fromMySQLInt32: Parameter error."

fromMySQLInt64U :: MySQLValue -> Int
fromMySQLInt64U (MySQLInt64U a) = read (show a) :: Int
fromMySQLInt64U _ = error "fromMySQLInt64U: Parameter error."

fromMySQLInt64 :: MySQLValue -> Int
fromMySQLInt64 (MySQLInt64 a) = read (show a) :: Int
fromMySQLInt64 _ = error "fromMySQLInt64: Parameter error."

-- Here only consider decimal without fractional.
fromMySQLDecimal :: MySQLValue -> Double
fromMySQLDecimal (MySQLDecimal a) = read (show a) :: Double
fromMySQLDecimal _ = error "fromMySQLDecimal: Parameter error."

fromMySQLText :: MySQLValue -> String
fromMySQLText (MySQLText a)
    | str!!0 == '\"' = DL.init $ DL.tail str     -- Throw away char " at head and rear.
    | otherwise = str
    where
      str = DT.unpack a
fromMySQLText _ = error "fromMySQLText: Parameter error."

fromMySQLNullText :: MySQLValue -> String
fromMySQLNullText MySQLNull = ""
fromMySQLNullText _ = error "fromMySQLNullText: Parameter error."

fromMySQLNullVarchar :: MySQLValue -> String
fromMySQLNullVarchar MySQLNull = ""
fromMySQLNullVarchar _ = error "fromMySQLNullVarchar: Parameter error."

fromMySQLFloat :: MySQLValue -> Float
fromMySQLFloat (MySQLFloat a) = read (show a) :: Float
fromMySQLFloat _ = error "fromMySQLFloat: Parameter error."

fromMySQLDouble :: MySQLValue -> Double
fromMySQLDouble (MySQLDouble a) = read (show a) :: Double
fromMySQLDouble _ = error "fromMySQLDouble: Parameter error."

toMySQLInt8 :: Int -> MySQLValue
toMySQLInt8 v = MySQLInt8 (read (show v) :: Int8)

toMySQLInt16U :: Int -> MySQLValue
toMySQLInt16U v = MySQLInt16U (read (show v) :: Word16)

toMySQLInt32U :: Int -> MySQLValue
toMySQLInt32U v = MySQLInt32U (read (show v) :: Word32)

toMySQLInt32 :: Int -> MySQLValue
toMySQLInt32 v = MySQLInt32 (read (show v) :: Int32)

toMySQLInt64U :: Int -> MySQLValue
toMySQLInt64U v = MySQLInt64U (read (show v) :: Word64)

toMySQLInt64 :: Int -> MySQLValue
toMySQLInt64 v = MySQLInt64 (read (show v) :: Int64)

toMySQLDecimal :: Int -> MySQLValue
toMySQLDecimal v = MySQLDecimal (read (show v) :: Scientific)

toMySQLBytes :: String -> MySQLValue
toMySQLBytes v = MySQLBytes (BC.pack v)                                         -- Convert String to ByteString

toMySQLText :: String -> MySQLValue
toMySQLText v = MySQLText (DT.pack v)

toMySQLNullText :: MySQLValue
toMySQLNullText = MySQLNull

toMySQLNullVarchar :: MySQLValue
toMySQLNullVarchar = MySQLNull

toMySQLFloat :: Float -> MySQLValue
toMySQLFloat v = MySQLFloat (read (show v) :: Float)

toMySQLDouble :: Double -> MySQLValue
toMySQLDouble v = MySQLDouble (read (show v) :: Double)

getColumnDB :: ColumnDef -> ByteString
getColumnDB (ColumnDef db _ _ _ _ _ _ _ _ _) = read (show db) :: ByteString

getColumnTable :: ColumnDef -> ByteString
getColumnTable (ColumnDef _ tb _ _ _ _ _ _ _ _) = read (show tb) :: ByteString

getColumnOrigTable :: ColumnDef -> ByteString
getColumnOrigTable (ColumnDef _ _ ot _ _ _ _ _ _ _) = read (show ot) :: ByteString

getColumnName :: ColumnDef -> ByteString
getColumnName (ColumnDef _ _ _ na _ _ _ _ _ _) = read (show na) :: ByteString

getColumnOrigName :: ColumnDef -> ByteString
getColumnOrigName (ColumnDef _ _ _ _ on _ _ _ _ _) = read (show on) :: ByteString

getColumnCharSet :: ColumnDef -> Word16
getColumnCharSet (ColumnDef _ _ _ _ _ cs _ _ _ _) = read (show cs) :: Word16

getColumnLength :: ColumnDef -> Word32
getColumnLength (ColumnDef _ _ _ _ _ _ ln _ _ _) = read (show ln) :: Word32

getColumnType :: ColumnDef -> String
getColumnType (ColumnDef _ _ _ _ _ _ _ tp _ _) = show tp

getColumnFlags :: ColumnDef -> Word16
getColumnFlags (ColumnDef _ _ _ _ _ _ _ _ fl _) = read (show fl) :: Word16

getColumnDecimals :: ColumnDef -> Word8
getColumnDecimals (ColumnDef _ _ _ _ _ _ _ _ _ de) = read (show de) :: Word8

getOkAffectedRows :: OK -> Int
getOkAffectedRows (OK okAffectedRow _ _ _) = okAffectedRow

getOkLastInsertID :: OK -> Int
getOkLastInsertID (OK _ okLastInsertID _ _) = okLastInsertID

getOkStatus :: OK -> Word16
getOkStatus (OK _ _ okStatus _) = okStatus

getOkWarningCnt :: OK -> Word16
getOkWarningCnt (OK _ _ _ okWarningCnt) = okWarningCnt

{- Read a value from input stream [MySQLValue], append it to existed string list, then read the next,
 - until read Nothing.
 -}
readStreamByText :: [String] -> S.InputStream [MySQLValue] -> IO [String]
readStreamByText es is = do
    S.read is >>= \x -> case x of                                       -- Dumb element 'case' is an array with type [MySQLValue]
        Just [MySQLText v] -> readStreamByText (es ++ [fromMySQLText (MySQLText v)]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed Double list, then read the next,
 - until read Nothing.
 - For every element of [MySQLValue], it is value list of Decimal.
 -}
readStreamByInt64 :: [Int] -> S.InputStream [MySQLValue] -> IO [Int]
readStreamByInt64 es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByInt64 (es ++ [fromMySQLInt64 (x!!0)]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed (Int8, Double) list, then read the next,
 - until read Nothing.
 - For every element of [MySQLValue], it is value list of tinyint and decimal.
 -}
readStreamByInt8Decimal :: [(Int,Double)] -> S.InputStream [MySQLValue] -> IO [(Int,Double)]
readStreamByInt8Decimal es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByInt8Decimal (es ++ [(fromMySQLInt8 (x!!0), fromMySQLDecimal (x!!1))]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed integer list, then read the next,
 - until read Nothing.
 - For every element of [MySQLValue], it is value list of tinyint and bigint.
 -}
readStreamByInt8Int64 :: [[Int]] -> S.InputStream [MySQLValue] -> IO [[Int]]
readStreamByInt8Int64 es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByInt8Int64 (es ++ [[fromMySQLInt8 (x!!0), fromMySQLInt64 (x!!1)]]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed list [(Int, String)], then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLInt32, MySQLText].
 -}
readStreamByInt32Text :: [(Int, String)] -> S.InputStream [MySQLValue] -> IO [(Int, String)]
readStreamByInt32Text es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByInt32Text (es ++ [(fromMySQLInt32 (x!!0), fromMySQLText (x!!1))]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed list [(Int, String)], then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLInt32U, MySQLText].
 -}
readStreamByInt32UText :: [(Int, String)] -> S.InputStream [MySQLValue] -> IO [(Int, String)]
readStreamByInt32UText es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByInt32UText (es ++ [(fromMySQLInt32U (x!!0), fromMySQLText (x!!1))]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed list [(Int, String, String)], then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLInt32U, MySQLText, MySQLText].
 -}
readStreamByInt32TextText :: [(Int, String, String)] -> S.InputStream [MySQLValue] -> IO [(Int, String, String)]
readStreamByInt32TextText es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByInt32TextText (es ++ [(fromMySQLInt32 (x!!0), fromMySQLText (x!!1), fromMySQLText (x!!2))]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed list [(Int, String, String, String)], then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLInt32U, MySQLText, MySQLText, MySqlText].
 -}
readStreamByInt32UTextTextText :: [(Int, String, String, String)] -> S.InputStream [MySQLValue] -> IO [(Int, String, String, String)]
readStreamByInt32UTextTextText es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByInt32UTextTextText (es ++ [(fromMySQLInt32U (x!!0), fromMySQLText (x!!1), fromMySQLText (x!!2), fromMySQLText (x!!3))]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed list [(String, String)], then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLText, MySQLText].
 -}
readStreamByTextText :: [(String, String)] -> S.InputStream [MySQLValue] -> IO [(String, String)]
readStreamByTextText es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByTextText (es ++ [(fromMySQLText (x!!0), fromMySQLText (x!!1))]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed list [(String, String, String, String)], then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLText, MySQLText, MySQLText, MySQLText].
 -}
readStreamByTextTextTextText :: [(String, String, String, String)] -> S.InputStream [MySQLValue] -> IO [(String, String, String, String)]
readStreamByTextTextTextText es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByTextTextTextText (es ++ [(fromMySQLText (x!!0), fromMySQLText (x!!1), fromMySQLText (x!!2), fromMySQLText (x!!3))]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed list [(Int, (String, String)], then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLInt32U, MySQLText, MySQLText].
 -}
readStreamByInt32UTextText :: [(Int, (String, String))] -> S.InputStream [MySQLValue] -> IO [(Int, (String, String))]
readStreamByInt32UTextText es is =
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByInt32UTextText (es ++ [(fromMySQLInt32U (x!!0), (fromMySQLText (x!!1), fromMySQLText (x!!2)))]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed list [(Int, (String, String, String, String, Int)], then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLInt32U, MySQLText, MySQLText, MySQLText, MySQLText, MySQLInt8].
 -}
readStreamByInt32UTextTextTextTextInt8 :: [(Int, (String, String, String, String, Int))] -> S.InputStream [MySQLValue] -> IO [(Int, (String, String, String, String, Int))]
readStreamByInt32UTextTextTextTextInt8 es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByInt32UTextTextTextTextInt8 (es ++ [(fromMySQLInt32U (x!!0), (fromMySQLText (x!!1), fromMySQLText (x!!2), fromMySQLText (x!!3), fromMySQLText (x!!4), fromMySQLInt8 (x!!5)))]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed string list, then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLText, MySQLText, MySQLInt8].
 -}
readStreamByTextTextInt8 :: [String] -> S.InputStream [MySQLValue] -> IO [String]
readStreamByTextTextInt8 es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByTextTextInt8 (es ++ [fromMySQLText (x!!0) ++ "_" ++ fromMySQLText (x!!1) ++ "_" ++ show (fromMySQLInt8 (x!!2))]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed string list, then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLText, MySQLText, MySQLInt8, MySQLText].
 -}
readStreamByTextTextInt8Text :: [String] -> S.InputStream [MySQLValue] -> IO [String]
readStreamByTextTextInt8Text es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByTextTextInt8Text (es ++ [fromMySQLText (x!!0) ++ "_" ++ fromMySQLText (x!!1) ++ "_" ++ show (fromMySQLInt8 (x!!2)) ++ "_" ++ fromMySQLText (x!!3)]) is
        Nothing -> return es

{- Read a value from input stream [MySQLValue], append it to existed string list, then read the next,
 - until read Nothing.
 - Here [MySQLValue] is [MySQLText, MySQLText, MySQLDouble].
 -}
readStreamByTextTextDouble :: [(String, String, Double)] -> S.InputStream [MySQLValue] -> IO [(String, String, Double)]
readStreamByTextTextDouble es is = do
    S.read is >>= \x -> case x of                                        -- Dumb element 'case' is an array with type [MySQLValue]
        Just x -> readStreamByTextTextDouble (es ++ [(fromMySQLText (x!!0), fromMySQLText (x!!1),  fromMySQLDouble (x!!2))]) is
        Nothing -> return es

-- Get a connection to MySQL database according to a configuration file.
getConn :: IO MySQLConn
getConn = do
    confInfo <- readFile "Configuration"                                        -- Read the local configuration file
    let host = getConfProperty "Host" confInfo
    let user = getConfProperty "User" confInfo
    let password = getConfProperty "Password" confInfo
    let database = getConfProperty "Database" confInfo
--  putStrLn $ "host:" ++ host ++ ", user:" ++ user ++ ", password:" ++ password ++ ", database:" ++ database

    connect defaultConnectInfo {
      ciHost = host,
      ciUser = BC.pack user,             -- Change String to ByteString
      ciPassword = BC.pack password,
      ciDatabase = BC.pack database
    }

-- Get a connection to database 'ccg4c' with user 'wqj'.
getConnByUserWqj :: IO MySQLConn
getConnByUserWqj = connect defaultConnectInfo {
    ciHost = "127.0.0.1",
    ciUser = "wqj",
    ciPassword = "wqj",
    ciDatabase = "ccg4c"
    }

-- Recognize type OK in module mysql-haskell.
recogOk :: IO ()
recogOk = do
    conn <- getConnByUserWqj
    let sqlstat = DS.fromString "create table if not exists test (id int, name char(8))"
    stmt <- prepareStmt conn sqlstat
    ok <- executeStmt conn stmt []
    putStrLn "recogOk: create table test (id int, name char[8])."
    putStrLn $ "recogOk: okAffectedRows = " ++ show (getOkAffectedRows ok)
    putStrLn $ "recogOk: okLastInsertID = " ++ show (getOkLastInsertID ok)
    putStrLn $ "recogOk: okStatus = " ++ show (getOkStatus ok)
    putStrLn $ "recogOk: okWarningCnt = " ++ show (getOkWarningCnt ok)

    let sqlstat = DS.fromString $ "insert into test set id = ?, name = ?"
    stmt <- prepareStmt conn sqlstat
    ok <- executeStmt conn stmt [toMySQLInt32 1, toMySQLText "QJWang"]
    putStrLn "recogOk: insert into test set id=1, name=\"QJWang\""
    putStrLn $ "recogOk: okAffectedRows = " ++ show (getOkAffectedRows ok)
    putStrLn $ "recogOk: okLastInsertID = " ++ show (getOkLastInsertID ok)
    putStrLn $ "recogOk: okStatus = " ++ show (getOkStatus ok)
    putStrLn $ "recogOk: okWarningCnt = " ++ show (getOkWarningCnt ok)

    let sqlstat = DS.fromString $ "insert into test set id = ?, name = ?"
    stmt <- prepareStmt conn sqlstat
    ok <- executeStmt conn stmt [toMySQLInt32 2, toMySQLText "HSSC"]
    putStrLn "recogOk: insert into test set id=2, name=\"HSSC\""
    putStrLn $ "recogOk: okAffectedRows = " ++ show (getOkAffectedRows ok)
    putStrLn $ "recogOk: okLastInsertID = " ++ show (getOkLastInsertID ok)
    putStrLn $ "recogOk: okStatus = " ++ show (getOkStatus ok)
    putStrLn $ "recogOk: okWarningCnt = " ++ show (getOkWarningCnt ok)

    let sqlstat = DS.fromString $ "update test set name = ? where id = ?"
    stmt <- prepareStmt conn sqlstat
    ok <- executeStmt conn stmt [toMySQLText "HuaShui", toMySQLInt32 1]
    putStrLn "recogOk: update test set name=\"HuaShui\" where id = 1"
    putStrLn $ "recogOk: okAffectedRows = " ++ show (getOkAffectedRows ok)
    putStrLn $ "recogOk: okLastInsertID = " ++ show (getOkLastInsertID ok)
    putStrLn $ "recogOk: okStatus = " ++ show (getOkStatus ok)
    putStrLn $ "recogOk: okWarningCnt = " ++ show (getOkWarningCnt ok)

    let sqlstat = DS.fromString $ "update test set name = ? where id = ?"
    stmt <- prepareStmt conn sqlstat
    ok <- executeStmt conn stmt [toMySQLText "HSSC", toMySQLInt32 2]
    putStrLn "recogOk: update test set name=\"HSSC\" where id = 2"
    putStrLn $ "recogOk: okAffectedRows = " ++ show (getOkAffectedRows ok)
    putStrLn $ "recogOk: okLastInsertID = " ++ show (getOkLastInsertID ok)
    putStrLn $ "recogOk: okStatus = " ++ show (getOkStatus ok)
    putStrLn $ "recogOk: okWarningCnt = " ++ show (getOkWarningCnt ok)

-- ================= Read word embedding file into MySQL table ===================

-- 解析一行：word f1 f2 ... f300
-- 返回 (word文本, 向量数组JSON文本)
parseLine :: Text -> Maybe (Text, Text)
parseLine line =
  let parts = DT.words line
  in if length parts == 301
     then
       let w = head parts
           rawFs = tail parts
           mFs = traverse readDouble rawFs
       in case mFs of
            Just fs ->
              let jsonBS = encode fs   -- 直接生成 [f1, f2, ...] JSON数组
                  jsonTxt = decodeUtf8 (toStrict jsonBS)  -- ✅ LazyBS → StrictBS → Text
              in Just (w, jsonTxt)
            Nothing -> Nothing
     else Nothing
  where
    readDouble :: Text -> Maybe Double
    readDouble t = case reads (DT.unpack t) of
      [(n,"")] -> Just n
      _ -> Nothing

-- ======== 把词嵌入文件导入数据库 =======
wordEmbedFile2MySQL :: IO ()
wordEmbedFile2MySQL = do
  putStr "向量文本文件路径："    -- D:\GitHub\wordEmb\sgns_renmin_word
  fp <- SIO.getLine
  putStr "数据库表："           -- sgns_renmin_word
  tblName <- SIO.getLine
  putStr "word字段名："         -- "word"
  colWord <- SIO.getLine
  putStr "emb字段名："          -- "emb"
  colEmb <- SIO.getLine
  conn <- getConn
  let sqlstat = DS.fromString $ "INSERT IGNORE INTO " ++ tblName ++ "(" ++ colWord ++ "," ++ colEmb ++ ") VALUES (?,?)"
  stmt <- prepareStmt conn sqlstat

  h <- SIO.openFile fp ReadMode
  loop conn stmt h 0
  SIO.hClose h
  closeStmt conn stmt
  close conn
  putStrLn "✅ 导入完成"
  where
    loop :: MySQLConn -> StmtID -> Handle -> Int -> IO ()
    loop c s h cnt = do
      e <- hIsEOF h
      if e
        then putStrLn $ "成功导入总行数：" ++ show cnt
        else do
          lineStr <- SIO.hGetLine h
          let l = DT.pack lineStr                            -- String → Text
          case parseLine l of
            Just (w, embTxt) -> do
              executeStmt c s [ MySQLText w, MySQLText embTxt ]
              loop c s h (cnt + 1)
            Nothing -> do
              putStrLn $ "⚠️ 跳过坏行：" ++ DT.unpack (DT.take 100 l)
              loop c s h cnt
