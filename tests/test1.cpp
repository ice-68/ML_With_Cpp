#include <gtest/gtest.h>
#include "tensor.h"

TEST(TensorTest, ScalarCreation) {
    Tensor t(3.14f);
    EXPECT_FLOAT_EQ(t.item(), 3.14f);
}

TEST(TensorTest, 1DCreation) {
    Tensor t(std::vector<float>{1.0f, 2.0f, 3.0f});

    ASSERT_EQ(t.shape().size(), 1);
    EXPECT_EQ(t.shape()[0], 3);

    EXPECT_FLOAT_EQ(t(0), 1.0f);
    EXPECT_FLOAT_EQ(t(1), 2.0f);
    EXPECT_FLOAT_EQ(t(2), 3.0f);
}

TEST(TensorTest, 2DCreation) {
    Tensor t(std::vector<std::vector<float>>{
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    });

    ASSERT_EQ(t.shape().size(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 2);

    EXPECT_FLOAT_EQ(t(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(t(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(t(1, 1), 4.0f);
}

TEST(TensorTest, MutableAccess) {
    Tensor t(std::vector<float>{1.0f, 2.0f, 3.0f});

    t(1) = 99.0f;
    EXPECT_FLOAT_EQ(t(1), 99.0f);
}

TEST(TensorTest, PrintScalar) {
    Tensor t(42.0f);
    testing::internal::CaptureStdout();
    std::cout << t;
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "42");
}

TEST(TensorTest, Print1D) {
    Tensor t(std::vector<float>{1.0f, 2.0f, 3.0f});
    testing::internal::CaptureStdout();
    std::cout << t;
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "[1.000000, 2.000000, 3.000000]");
}

TEST(TensorTest, OutOfBounds1D) {
    Tensor t(std::vector<float>{1.0f, 2.0f});
    EXPECT_THROW(t(5), std::invalid_argument);
}

TEST(TensorTest, OutOfBounds2D) {
    Tensor t(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    EXPECT_THROW(t(0, 5), std::invalid_argument);
    EXPECT_THROW(t(5, 0), std::invalid_argument);
}
