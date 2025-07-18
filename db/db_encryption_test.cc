//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
#include <iostream>
#include <string>

#include "db/db_test_util.h"
#include "port/stack_trace.h"
#include "rocksdb/perf_context.h"
#include "test_util/sync_point.h"

namespace ROCKSDB_NAMESPACE {

class DBEncryptionTest : public DBTestBase {
 public:
  DBEncryptionTest()
      : DBTestBase("db_encryption_test", /*env_do_fsync=*/true) {}
  Env* GetNonEncryptedEnv() {
    if (encrypted_env_ != nullptr) {
      return (static_cast_with_check<CompositeEnvWrapper>(encrypted_env_))
          ->env_target();
    } else {
      return env_;
    }
  }
};

TEST_F(DBEncryptionTest, CheckEncrypted) {
  ASSERT_OK(Put("foo567", "v1.fetdq"));
  ASSERT_OK(Put("bar123", "v2.dfgkjdfghsd"));
  Close();

  // Open all files and look for the values we've put in there.
  // They should not be found if encrypted, otherwise
  // they should be found.
  std::vector<std::string> fileNames;
  auto status = env_->GetChildren(dbname_, &fileNames);
  ASSERT_OK(status);

  Env* target = GetNonEncryptedEnv();
  int hits = 0;
  for (auto it = fileNames.begin(); it != fileNames.end(); ++it) {
    if (*it == "LOCK") {
      continue;
    }
    auto filePath = dbname_ + "/" + *it;
    std::unique_ptr<SequentialFile> seqFile;
    auto envOptions = EnvOptions(CurrentOptions());
    status = target->NewSequentialFile(filePath, &seqFile, envOptions);
    ASSERT_OK(status);

    uint64_t fileSize;
    status = target->GetFileSize(filePath, &fileSize);
    ASSERT_OK(status);

    std::string scratch;
    scratch.reserve(fileSize);
    Slice data;
    status = seqFile->Read(fileSize, &data, (char*)scratch.data());
    ASSERT_OK(status);

    if (data.ToString().find("foo567") != std::string::npos) {
      hits++;
      // std::cout << "Hit in " << filePath << "\n";
    }
    if (data.ToString().find("v1.fetdq") != std::string::npos) {
      hits++;
      // std::cout << "Hit in " << filePath << "\n";
    }
    if (data.ToString().find("bar123") != std::string::npos) {
      hits++;
      // std::cout << "Hit in " << filePath << "\n";
    }
    if (data.ToString().find("v2.dfgkjdfghsd") != std::string::npos) {
      hits++;
      // std::cout << "Hit in " << filePath << "\n";
    }
    if (data.ToString().find("dfgk") != std::string::npos) {
      hits++;
      // std::cout << "Hit in " << filePath << "\n";
    }
  }
  if (encrypted_env_) {
    ASSERT_EQ(hits, 0);
  } else {
    ASSERT_GE(hits, 4);
  }
}

TEST_F(DBEncryptionTest, ReadEmptyFile) {
  auto defaultEnv = GetNonEncryptedEnv();

  // create empty file for reading it back in later
  auto envOptions = EnvOptions(CurrentOptions());
  auto filePath = dbname_ + "/empty.empty";

  Status status;
  {
    std::unique_ptr<WritableFile> writableFile;
    status = defaultEnv->NewWritableFile(filePath, &writableFile, envOptions);
    ASSERT_OK(status);
  }

  std::unique_ptr<SequentialFile> seqFile;
  status = defaultEnv->NewSequentialFile(filePath, &seqFile, envOptions);
  ASSERT_OK(status);

  std::string scratch;
  Slice data;
  // reading back 16 bytes from the empty file shouldn't trigger an assertion.
  // it should just work and return an empty string
  status = seqFile->Read(16, &data, (char*)scratch.data());
  ASSERT_OK(status);

  ASSERT_TRUE(data.empty());
}

TEST_F(DBEncryptionTest, NotSupportedGetFileSize) {
  // Validate envrypted env does not support GetFileSize.
  // The goal of the test is to validate the encrypted env/fs does not support
  // GetFileSize API on FSRandomAccessFile interface.
  // This test combined with the rest of the integration tests validate that
  // the new API GetFileSize on FSRandomAccessFile interface is not required to
  // be supported for database to work properly.
  // The GetFileSize API is used in ReadFooterFromFile() API to get the file
  // size. When GetFileSize API is not supported, the ReadFooterFromFile() API
  // will use FileSystem GetFileSize API as fallback. Refer to the
  // EncryptedRandomAccessFile class definition for more details.
  if (!encrypted_env_) {
    return;
  }

  auto fs = encrypted_env_->GetFileSystem();

  // create empty file for reading it back in later
  auto filePath = dbname_ + "/empty.empty";

  // Create empty file
  CreateFile(fs.get(), filePath, "", false);

  // Open it for reading footer
  std::unique_ptr<FSRandomAccessFile> randomAccessFile;
  auto status = fs->NewRandomAccessFile(filePath, FileOptions(),
                                        &randomAccessFile, nullptr);
  ASSERT_OK(status);

  uint64_t fileSize;
  status = randomAccessFile->GetFileSize(&fileSize);
  ASSERT_TRUE(status.IsNotSupported());
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
