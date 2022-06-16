//  Copyright (c) 2020-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/blob/blob_index.h"
#include "db/db_test_util.h"
#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

enum class WriteBatchOpType {
  kPut = 0,
  kDelete,
  kSingleDelete,
  kDeleteRange,
  kMerge,
  kBlobIndex,
  kNum,
};

// Integer addition is needed for `::testing::Range()` to take the enum type.
WriteBatchOpType operator+(WriteBatchOpType lhs, const int rhs) {
  using T = std::underlying_type<WriteBatchOpType>::type;
  return static_cast<WriteBatchOpType>(static_cast<T>(lhs) + rhs);
}

std::pair<WriteBatch, Status> GetWriteBatch(ColumnFamilyHandle* cf_handle,
                                            WriteBatchOpType op_type) {
  Status s;
  WriteBatch wb(0 /* reserved_bytes */, 0 /* max_bytes */,
                8 /* protection_bytes_per_entry */, 0 /* default_cf_ts_sz */);
  switch (op_type) {
    case WriteBatchOpType::kPut:
      s = wb.Put(cf_handle, "key", "val");
      break;
    case WriteBatchOpType::kDelete:
      s = wb.Delete(cf_handle, "key");
      break;
    case WriteBatchOpType::kSingleDelete:
      s = wb.SingleDelete(cf_handle, "key");
      break;
    case WriteBatchOpType::kDeleteRange:
      s = wb.DeleteRange(cf_handle, "begin", "end");
      break;
    case WriteBatchOpType::kMerge:
      s = wb.Merge(cf_handle, "key", "val");
      break;
    case WriteBatchOpType::kBlobIndex: {
      // TODO(ajkr): use public API once available.
      uint32_t cf_id;
      if (cf_handle == nullptr) {
        cf_id = 0;
      } else {
        cf_id = cf_handle->GetID();
      }

      std::string blob_index;
      BlobIndex::EncodeInlinedTTL(&blob_index, /* expiration */ 9876543210,
                                  "val");

      s = WriteBatchInternal::PutBlobIndex(&wb, cf_id, "key", blob_index);
      break;
    }
    case WriteBatchOpType::kNum:
      assert(false);
  }
  return {std::move(wb), std::move(s)};
}

class DbKvChecksumTest
    : public DBTestBase,
      public ::testing::WithParamInterface<std::tuple<WriteBatchOpType, char>> {
 public:
  DbKvChecksumTest()
      : DBTestBase("db_kv_checksum_test", /*env_do_fsync=*/false) {
    op_type_ = std::get<0>(GetParam());
    corrupt_byte_addend_ = std::get<1>(GetParam());
  }

  void CorruptNextByteCallBack(void* arg) {
    Slice encoded = *static_cast<Slice*>(arg);
    if (entry_len_ == std::numeric_limits<size_t>::max()) {
      // We learn the entry size on the first attempt
      entry_len_ = encoded.size();
    }
    // All entries should be the same size
    assert(entry_len_ == encoded.size());
    char* buf = const_cast<char*>(encoded.data());
    buf[corrupt_byte_offset_] += corrupt_byte_addend_;
    ++corrupt_byte_offset_;
  }

  bool MoreBytesToCorrupt() { return corrupt_byte_offset_ < entry_len_; }

 protected:
  WriteBatchOpType op_type_;
  char corrupt_byte_addend_;
  size_t corrupt_byte_offset_ = 0;
  size_t entry_len_ = std::numeric_limits<size_t>::max();
};

std::string GetOpTypeString(const WriteBatchOpType& op_type) {
  switch (op_type) {
    case WriteBatchOpType::kPut:
      return "Put";
    case WriteBatchOpType::kDelete:
      return "Delete";
    case WriteBatchOpType::kSingleDelete:
      return "SingleDelete";
    case WriteBatchOpType::kDeleteRange:
      return "DeleteRange";
      break;
    case WriteBatchOpType::kMerge:
      return "Merge";
      break;
    case WriteBatchOpType::kBlobIndex:
      return "BlobIndex";
      break;
    case WriteBatchOpType::kNum:
      assert(false);
  }
  assert(false);
  return "";
}

INSTANTIATE_TEST_CASE_P(
    DbKvChecksumTest, DbKvChecksumTest,
    ::testing::Combine(::testing::Range(static_cast<WriteBatchOpType>(0),
                                        WriteBatchOpType::kNum),
                       ::testing::Values(2, 103, 251)),
    [](const testing::TestParamInfo<std::tuple<WriteBatchOpType, char>>& args) {
      std::ostringstream oss;
      oss << GetOpTypeString(std::get<0>(args.param)) << "Add"
          << static_cast<int>(
                 static_cast<unsigned char>(std::get<1>(args.param)));
      return oss.str();
    });

TEST_P(DbKvChecksumTest, MemTableAddCorrupted) {
  // This test repeatedly attempts to write `WriteBatch`es containing a single
  // entry of type `op_type_`. Each attempt has one byte corrupted in its
  // memtable entry by adding `corrupt_byte_addend_` to its original value. The
  // test repeats until an attempt has been made on each byte in the encoded
  // memtable entry. All attempts are expected to fail with `Status::Corruption`
  SyncPoint::GetInstance()->SetCallBack(
      "MemTable::Add:Encoded",
      std::bind(&DbKvChecksumTest::CorruptNextByteCallBack, this,
                std::placeholders::_1));

  while (MoreBytesToCorrupt()) {
    // Failed memtable insert always leads to read-only mode, so we have to
    // reopen for every attempt.
    Options options = CurrentOptions();
    if (op_type_ == WriteBatchOpType::kMerge) {
      options.merge_operator = MergeOperators::CreateStringAppendOperator();
    }
    Reopen(options);

    SyncPoint::GetInstance()->EnableProcessing();
    auto batch_and_status = GetWriteBatch(nullptr /* cf_handle */, op_type_);
    ASSERT_OK(batch_and_status.second);
    ASSERT_TRUE(
        db_->Write(WriteOptions(), &batch_and_status.first).IsCorruption());
    SyncPoint::GetInstance()->DisableProcessing();

    // In case the above callback is not invoked, this test will run
    // numeric_limits<size_t>::max() times until it reports an error (or will
    // exhaust disk space). Added this assert to report error early.
    ASSERT_TRUE(entry_len_ < std::numeric_limits<size_t>::max());
  }
}

TEST_P(DbKvChecksumTest, MemTableAddWithColumnFamilyCorrupted) {
  // This test repeatedly attempts to write `WriteBatch`es containing a single
  // entry of type `op_type_` to a non-default column family. Each attempt has
  // one byte corrupted in its memtable entry by adding `corrupt_byte_addend_`
  // to its original value. The test repeats until an attempt has been made on
  // each byte in the encoded memtable entry. All attempts are expected to fail
  // with `Status::Corruption`.
  Options options = CurrentOptions();
  if (op_type_ == WriteBatchOpType::kMerge) {
    options.merge_operator = MergeOperators::CreateStringAppendOperator();
  }
  CreateAndReopenWithCF({"pikachu"}, options);
  SyncPoint::GetInstance()->SetCallBack(
      "MemTable::Add:Encoded",
      std::bind(&DbKvChecksumTest::CorruptNextByteCallBack, this,
                std::placeholders::_1));

  while (MoreBytesToCorrupt()) {
    // Failed memtable insert always leads to read-only mode, so we have to
    // reopen for every attempt.
    ReopenWithColumnFamilies({kDefaultColumnFamilyName, "pikachu"}, options);

    SyncPoint::GetInstance()->EnableProcessing();
    auto batch_and_status = GetWriteBatch(handles_[1], op_type_);
    ASSERT_OK(batch_and_status.second);
    ASSERT_TRUE(
        db_->Write(WriteOptions(), &batch_and_status.first).IsCorruption());
    SyncPoint::GetInstance()->DisableProcessing();

    // In case the above callback is not invoked, this test will run
    // numeric_limits<size_t>::max() times until it reports an error (or will
    // exhaust disk space). Added this assert to report error early.
    ASSERT_TRUE(entry_len_ < std::numeric_limits<size_t>::max());
  }
}

TEST_P(DbKvChecksumTest, NoCorruptionCase) {
  // If this test fails, we may have found a piece of malfunctioned hardware
  auto batch_and_status = GetWriteBatch(nullptr, op_type_);
  ASSERT_OK(batch_and_status.second);
  ASSERT_OK(batch_and_status.first.VerifyChecksum());
}

TEST_P(DbKvChecksumTest, WriteToWALCorrupted) {
  // This test repeatedly attempts to write `WriteBatch`es containing a single
  // entry of type `op_type_`. Each attempt has one byte corrupted by adding
  // `corrupt_byte_addend_` to its original value. The test repeats until an
  // attempt has been made on each byte in the encoded write batch. All attempts
  // are expected to fail with `Status::Corruption`
  Options options = CurrentOptions();
  if (op_type_ == WriteBatchOpType::kMerge) {
    options.merge_operator = MergeOperators::CreateStringAppendOperator();
  }
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry",
      std::bind(&DbKvChecksumTest::CorruptNextByteCallBack, this,
                std::placeholders::_1));
  // First 8 bytes are for sequence number which is not protected in write batch
  corrupt_byte_offset_ = 8;

  while (MoreBytesToCorrupt()) {
    // Corrupted write batch leads to read-only mode, so we have to
    // reopen for every attempt.
    Reopen(options);
    auto log_size_pre_write = dbfull()->TEST_total_log_size();

    SyncPoint::GetInstance()->EnableProcessing();
    auto batch_and_status = GetWriteBatch(nullptr /* cf_handle */, op_type_);
    ASSERT_OK(batch_and_status.second);
    ASSERT_TRUE(
        db_->Write(WriteOptions(), &batch_and_status.first).IsCorruption());
    // Confirm that nothing was written to WAL
    ASSERT_EQ(log_size_pre_write, dbfull()->TEST_total_log_size());
    ASSERT_TRUE(dbfull()->TEST_GetBGError().IsCorruption());
    SyncPoint::GetInstance()->DisableProcessing();

    // In case the above callback is not invoked, this test will run
    // numeric_limits<size_t>::max() times until it reports an error (or will
    // exhaust disk space). Added this assert to report error early.
    ASSERT_TRUE(entry_len_ < std::numeric_limits<size_t>::max());
  }
}

TEST_P(DbKvChecksumTest, WriteToWALWithColumnFamilyCorrupted) {
  // This test repeatedly attempts to write `WriteBatch`es containing a single
  // entry of type `op_type_`. Each attempt has one byte corrupted by adding
  // `corrupt_byte_addend_` to its original value. The test repeats until an
  // attempt has been made on each byte in the encoded write batch. All attempts
  // are expected to fail with `Status::Corruption`
  Options options = CurrentOptions();
  if (op_type_ == WriteBatchOpType::kMerge) {
    options.merge_operator = MergeOperators::CreateStringAppendOperator();
  }
  CreateAndReopenWithCF({"pikachu"}, options);
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry",
      std::bind(&DbKvChecksumTest::CorruptNextByteCallBack, this,
                std::placeholders::_1));
  // First 8 bytes are for sequence number which is not protected in write batch
  corrupt_byte_offset_ = 8;

  while (MoreBytesToCorrupt()) {
    // Corrupted write batch leads to read-only mode, so we have to
    // reopen for every attempt.
    ReopenWithColumnFamilies({kDefaultColumnFamilyName, "pikachu"}, options);
    auto log_size_pre_write = dbfull()->TEST_total_log_size();

    SyncPoint::GetInstance()->EnableProcessing();
    auto batch_and_status = GetWriteBatch(handles_[1], op_type_);
    ASSERT_OK(batch_and_status.second);
    ASSERT_TRUE(
        db_->Write(WriteOptions(), &batch_and_status.first).IsCorruption());
    // Confirm that nothing was written to WAL
    ASSERT_EQ(log_size_pre_write, dbfull()->TEST_total_log_size());
    ASSERT_TRUE(dbfull()->TEST_GetBGError().IsCorruption());
    SyncPoint::GetInstance()->DisableProcessing();

    // In case the above callback is not invoked, this test will run
    // numeric_limits<size_t>::max() times until it reports an error (or will
    // exhaust disk space). Added this assert to report error early.
    ASSERT_TRUE(entry_len_ < std::numeric_limits<size_t>::max());
  }
}

class DbKvChecksumTestMergedBatch
    : public DBTestBase,
      public ::testing::WithParamInterface<
          std::tuple<WriteBatchOpType, WriteBatchOpType, char>> {
 public:
  DbKvChecksumTestMergedBatch()
      : DBTestBase("db_kv_checksum_test", /*env_do_fsync=*/false) {
    op_type1_ = std::get<0>(GetParam());
    op_type2_ = std::get<1>(GetParam());
    corrupt_byte_addend_ = std::get<2>(GetParam());
  }

 protected:
  WriteBatchOpType op_type1_;
  WriteBatchOpType op_type2_;
  char corrupt_byte_addend_;
};

void CorruptWriteBatch(Slice* content, size_t offset,
                       char corrupt_byte_addend) {
  ASSERT_TRUE(offset < content->size());
  char* buf = const_cast<char*>(content->data());
  buf[offset] += corrupt_byte_addend;
}

TEST_P(DbKvChecksumTestMergedBatch, NoCorruptionCase) {
  // Veirfy write batch checksum after write batch append
  auto batch1 = GetWriteBatch(nullptr /* cf_handle */, op_type1_);
  ASSERT_OK(batch1.second);
  auto batch2 = GetWriteBatch(nullptr /* cf_handle */, op_type2_);
  ASSERT_OK(batch2.second);
  ASSERT_OK(WriteBatchInternal::Append(&batch1.first, &batch2.first));
  ASSERT_OK(batch1.first.VerifyChecksum());
}

TEST_P(DbKvChecksumTestMergedBatch, WriteToWALCorrupted) {
  // This test has two writers repeatedly attempt to write `WriteBatch`es
  // containing a single entry of type op_type1_ and op_type2_ respectively. The
  // leader of the write group writes the batch containinng the entry of type
  // op_type1_. One byte of the pre-merged write batches is corrupted by adding
  // `corrupt_byte_addend_` to the batch's original value during each attempt.
  // The test repeats until an attempt has been made on each byte in both
  // pre-merged write batches. All attempts are expected to fail with
  // `Status::Corruption`.
  Options options = CurrentOptions();
  if (op_type1_ == WriteBatchOpType::kMerge ||
      op_type2_ == WriteBatchOpType::kMerge) {
    options.merge_operator = MergeOperators::CreateStringAppendOperator();
  }

  auto leader_batch_and_status =
      GetWriteBatch(nullptr /* cf_handle */, op_type1_);
  ASSERT_OK(leader_batch_and_status.second);
  auto follower_batch_and_status =
      GetWriteBatch(nullptr /* cf_handle */, op_type2_);
  size_t leader_batch_size = leader_batch_and_status.first.GetDataSize();
  size_t total_bytes =
      leader_batch_size + follower_batch_and_status.first.GetDataSize();
  // First 8 bytes are for sequence number which is not protected in write batch
  size_t corrupt_byte_offset = 8;

  std::atomic<bool> follower_joined{false};
  std::atomic<int> leader_count{0};
  port::Thread follower_thread;
  // This callback should only be called by the leader thread
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait2", [&](void* arg_leader) {
        auto* leader = reinterpret_cast<WriteThread::Writer*>(arg_leader);
        ASSERT_EQ(leader->state, WriteThread::STATE_GROUP_LEADER);

        // This callback should only be called by the follower thread
        SyncPoint::GetInstance()->SetCallBack(
            "WriteThread::JoinBatchGroup:Wait", [&](void* arg_follower) {
              auto* follower =
                  reinterpret_cast<WriteThread::Writer*>(arg_follower);
              // The leader thread will wait on this bool and hence wait until
              // this writer joins the write group
              ASSERT_NE(follower->state, WriteThread::STATE_GROUP_LEADER);
              if (corrupt_byte_offset >= leader_batch_size) {
                Slice batch_content = follower->batch->Data();
                CorruptWriteBatch(&batch_content,
                                  corrupt_byte_offset - leader_batch_size,
                                  corrupt_byte_addend_);
              }
              // Leader busy waits on this flag
              follower_joined = true;
              // So the follower does not enter the outer callback at
              // WriteThread::JoinBatchGroup:Wait2
              SyncPoint::GetInstance()->DisableProcessing();
            });

        // Start the other writer thread which will join the write group as
        // follower
        follower_thread = port::Thread([&]() {
          follower_batch_and_status =
              GetWriteBatch(nullptr /* cf_handle */, op_type2_);
          ASSERT_OK(follower_batch_and_status.second);
          ASSERT_TRUE(
              db_->Write(WriteOptions(), &follower_batch_and_status.first)
                  .IsCorruption());
        });

        ASSERT_EQ(leader->batch->GetDataSize(), leader_batch_size);
        if (corrupt_byte_offset < leader_batch_size) {
          Slice batch_content = leader->batch->Data();
          CorruptWriteBatch(&batch_content, corrupt_byte_offset,
                            corrupt_byte_addend_);
        }
        leader_count++;
        while (!follower_joined) {
          // busy waiting
        }
      });
  while (corrupt_byte_offset < total_bytes) {
    // Reopen DB since it failed WAL write which lead to read-only mode
    Reopen(options);
    SyncPoint::GetInstance()->EnableProcessing();
    auto log_size_pre_write = dbfull()->TEST_total_log_size();
    leader_batch_and_status = GetWriteBatch(nullptr /* cf_handle */, op_type1_);
    ASSERT_OK(leader_batch_and_status.second);
    ASSERT_TRUE(db_->Write(WriteOptions(), &leader_batch_and_status.first)
                    .IsCorruption());
    follower_thread.join();
    // Prevent leader thread from entering this callback
    SyncPoint::GetInstance()->ClearCallBack("WriteThread::JoinBatchGroup:Wait");
    ASSERT_EQ(1, leader_count);
    // Nothing should have been written to WAL
    ASSERT_EQ(log_size_pre_write, dbfull()->TEST_total_log_size());
    ASSERT_TRUE(dbfull()->TEST_GetBGError().IsCorruption());

    corrupt_byte_offset++;
    if (corrupt_byte_offset == leader_batch_size) {
      // skip over the sequence number part of follower's write batch
      corrupt_byte_offset += 8;
    }
    follower_joined = false;
    leader_count = 0;
  }
  SyncPoint::GetInstance()->DisableProcessing();
}

TEST_P(DbKvChecksumTestMergedBatch, WriteToWALWithColumnFamilyCorrupted) {
  // This test has two writers repeatedly attempt to write `WriteBatch`es
  // containing a single entry of type op_type1_ and op_type2_ respectively. The
  // leader of the write group writes the batch containinng the entry of type
  // op_type1_. One byte of the pre-merged write batches is corrupted by adding
  // `corrupt_byte_addend_` to the batch's original value during each attempt.
  // The test repeats until an attempt has been made on each byte in both
  // pre-merged write batches. All attempts are expected to fail with
  // `Status::Corruption`.
  Options options = CurrentOptions();
  if (op_type1_ == WriteBatchOpType::kMerge ||
      op_type2_ == WriteBatchOpType::kMerge) {
    options.merge_operator = MergeOperators::CreateStringAppendOperator();
  }
  CreateAndReopenWithCF({"ramen"}, options);

  auto leader_batch_and_status = GetWriteBatch(handles_[1], op_type1_);
  ASSERT_OK(leader_batch_and_status.second);
  auto follower_batch_and_status = GetWriteBatch(handles_[1], op_type2_);
  size_t leader_batch_size = leader_batch_and_status.first.GetDataSize();
  size_t total_bytes =
      leader_batch_size + follower_batch_and_status.first.GetDataSize();
  // First 8 bytes are for sequence number which is not protected in write batch
  size_t corrupt_byte_offset = 8;

  std::atomic<bool> follower_joined{false};
  std::atomic<int> leader_count{0};
  port::Thread follower_thread;
  // This callback should only be called by the leader thread
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait2", [&](void* arg_leader) {
        auto* leader = reinterpret_cast<WriteThread::Writer*>(arg_leader);
        ASSERT_EQ(leader->state, WriteThread::STATE_GROUP_LEADER);

        // This callback should only be called by the follower thread
        SyncPoint::GetInstance()->SetCallBack(
            "WriteThread::JoinBatchGroup:Wait", [&](void* arg_follower) {
              auto* follower =
                  reinterpret_cast<WriteThread::Writer*>(arg_follower);
              // The leader thread will wait on this bool and hence wait until
              // this writer joins the write group
              ASSERT_NE(follower->state, WriteThread::STATE_GROUP_LEADER);
              if (corrupt_byte_offset >= leader_batch_size) {
                Slice batch_content =
                    WriteBatchInternal::Contents(follower->batch);
                CorruptWriteBatch(&batch_content,
                                  corrupt_byte_offset - leader_batch_size,
                                  corrupt_byte_addend_);
              }
              follower_joined = true;
              // So the follower does not enter the outer callback at
              // WriteThread::JoinBatchGroup:Wait2
              SyncPoint::GetInstance()->DisableProcessing();
            });

        // Start the other writer thread which will join the write group as
        // follower
        follower_thread = port::Thread([&]() {
          follower_batch_and_status = GetWriteBatch(handles_[1], op_type2_);
          ASSERT_OK(follower_batch_and_status.second);
          ASSERT_TRUE(
              db_->Write(WriteOptions(), &follower_batch_and_status.first)
                  .IsCorruption());
        });

        ASSERT_EQ(leader->batch->GetDataSize(), leader_batch_size);
        if (corrupt_byte_offset < leader_batch_size) {
          Slice batch_content = WriteBatchInternal::Contents(leader->batch);
          CorruptWriteBatch(&batch_content, corrupt_byte_offset,
                            corrupt_byte_addend_);
        }
        leader_count++;
        while (!follower_joined) {
          // busy waiting
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();
  while (corrupt_byte_offset < total_bytes) {
    // Reopen DB since it failed WAL write which lead to read-only mode
    ReopenWithColumnFamilies({kDefaultColumnFamilyName, "ramen"}, options);
    SyncPoint::GetInstance()->EnableProcessing();
    auto log_size_pre_write = dbfull()->TEST_total_log_size();
    leader_batch_and_status = GetWriteBatch(handles_[1], op_type1_);
    ASSERT_OK(leader_batch_and_status.second);
    ASSERT_TRUE(db_->Write(WriteOptions(), &leader_batch_and_status.first)
                    .IsCorruption());
    follower_thread.join();
    // Prevent leader thread from entering this callback
    SyncPoint::GetInstance()->ClearCallBack("WriteThread::JoinBatchGroup:Wait");

    ASSERT_EQ(1, leader_count);
    // Nothing should have been written to WAL
    ASSERT_EQ(log_size_pre_write, dbfull()->TEST_total_log_size());
    ASSERT_TRUE(dbfull()->TEST_GetBGError().IsCorruption());

    corrupt_byte_offset++;
    if (corrupt_byte_offset == leader_batch_size) {
      // skip over the sequence number part of follower's write batch
      corrupt_byte_offset += 8;
    }
    follower_joined = false;
    leader_count = 0;
  }
  SyncPoint::GetInstance()->DisableProcessing();
}

INSTANTIATE_TEST_CASE_P(
    DbKvChecksumTestMergedBatch, DbKvChecksumTestMergedBatch,
    ::testing::Combine(::testing::Range(static_cast<WriteBatchOpType>(0),
                                        WriteBatchOpType::kNum),
                       ::testing::Range(static_cast<WriteBatchOpType>(0),
                                        WriteBatchOpType::kNum),
                       ::testing::Values(2, 103, 251)),
    [](const testing::TestParamInfo<
        std::tuple<WriteBatchOpType, WriteBatchOpType, char>>& args) {
      std::ostringstream oss;
      oss << GetOpTypeString(std::get<0>(args.param))
          << GetOpTypeString(std::get<1>(args.param)) << "Add"
          << static_cast<int>(
                 static_cast<unsigned char>(std::get<2>(args.param)));
      return oss.str();
    });

// TODO: add test for transactions
// TODO: add test for corrupted write batch with WAL disabled
}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
