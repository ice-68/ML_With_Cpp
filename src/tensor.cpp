#include "tensor.h"
#include <iostream>
#include <vector>
#include <string>

Tensor::Tensor(float data, bool requires_grad,
               std::function<void(const std::vector<float> &)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents)
    : _data{data}, _shape{}, _stride{}, _requires_grad{requires_grad}, _gradfn{gradfn}, _parents{parents}
{
    if (_requires_grad)
    {
        zero_grad();
    }
}

Tensor::Tensor(std::vector<float> data, bool requires_grad,
               std::function<void(const std::vector<float> &)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents)
    : _data{data}, _shape{data.size()}, _stride{1}, _requires_grad{requires_grad}, _gradfn{gradfn}, _parents{parents}
{
    if (_requires_grad)
    {
        zero_grad();
    }
}

Tensor::Tensor(std::vector<std::vector<float>> data, bool requires_grad,
               std::function<void(const std::vector<float> &)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents)
    : _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1}, _requires_grad{requires_grad}, _gradfn{gradfn}, _parents{parents}
{
    std::size_t n_expected_to_columns = data[0].size();
    for (std::size_t i = 0; i < data.size(); i++)
    {
        if (data[i].size() != n_expected_to_columns)
        {
            throw std::invalid_argument("Dimension is not consistent");
        }
    }
    for (std::size_t i = 0; i < data.size(); i++)
    {
        for (std::size_t j = 0; j < data[i].size(); j++)
        {
            _data.push_back(data[i][j]);
        }
    }
    if (_requires_grad)
    {
        zero_grad();
    }
}

const float &Tensor::item() const
{
    if (_data.size() == 1)
    {
        return _data[0];
    }
    throw std::runtime_error("item() can be only called on tensors with a single element");
}

float &Tensor::item()
{
    if (_data.size() == 1)
    {
        return _data[0];
    }
    throw std::runtime_error("item() can be only called on tensors with a single element");
}

const float &Tensor::operator()(std::size_t i) const
{
    if (_shape.size() == 0)
    {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if (_shape.size() == 1)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Index " + std::to_string(i) +
                                        " is out of bounds of array's size " +
                                        std::to_string(_shape[0]));
        }
        return _data[i];
    }
    throw std::invalid_argument("This is a 1D tensor. Use two indices for 2D tensors.");
}

float &Tensor::operator()(std::size_t i)
{
    if (_shape.size() == 0)
    {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if (_shape.size() == 1)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Index " + std::to_string(i) +
                                        " is out of bounds of array's size " +
                                        std::to_string(_shape[0]));
        }
        return _data[i];
    }
    throw std::invalid_argument("This is a 1D tensor. Use two indices for 2D tensors.");
}

const float &Tensor::operator()(std::size_t i, std::size_t j) const
{
    if (_shape.size() == 2)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Row Index " + std::to_string(i) +
                                        " is out of bounds of array's size " +
                                        std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1])
        {
            throw std::invalid_argument("Column Index " + std::to_string(j) +
                                        " is out of bounds of array's size " +
                                        std::to_string(_shape[1]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    throw std::invalid_argument("Can only double index into 2D tensor");
}

float &Tensor::operator()(std::size_t i, std::size_t j)
{
    if (_shape.size() == 2)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Row Index " + std::to_string(i) +
                                        " is out of bounds of array's size " +
                                        std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1])
        {
            throw std::invalid_argument("Column Index " + std::to_string(j) +
                                        " is out of bounds of array's size " +
                                        std::to_string(_shape[1]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    throw std::invalid_argument("Can only double index into 2D tensor");
}

std::ostream &operator<<(std::ostream &os, const Tensor &obj)
{
    std::string string_repr;
    if (obj.shape().size() == 0)
    {
        os << obj.item();
        return os;
    }
    else if (obj.shape().size() == 1)
    {
        string_repr += "[";
        for (std::size_t i = 0; i < obj.shape()[0]; i++)
        {
            string_repr += std::to_string(obj(i));
            if (i != obj.shape()[0] - 1)
            {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    else
    {
        for (std::size_t i = 0; i < obj.shape()[0]; i++)
        {
            string_repr += "[";
            for (std::size_t j = 0; j < obj.shape()[1]; j++)
            {
                string_repr += std::to_string(obj(i, j));
                if (j != obj.shape()[1] - 1)
                {
                    string_repr += ", ";
                }
            }
            string_repr += "]";
            if (i != obj.shape()[0] - 1)
            {
                string_repr += ", ";
            }
        }
    }
    os << string_repr;
    return os;
}

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other)
{
    // scalar + scalar
    if (_shape.size() == 0 && other->shape().size() == 0)
    {
        float result = item() + other->item();
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                self->add_to_grad({grad_output[0]});
                other->add_to_grad({grad_output[0]});
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }

    // scalar + 1D
    if (_shape.size() == 0 && other->shape().size() == 1)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < other->shape()[0]; i++)
        {
            result.push_back(item() + ((*other)(i)));
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                float grad_self = 0.0f;
                for (std::size_t i = 0; i < grad_output.size(); i++)
                {
                    grad_self += grad_output[i];
                }
                self->add_to_grad({grad_output[0]});
                other->add_to_grad({grad_output[0]});
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }

    // scalar + 2D

    if (_shape.size() == 0 && other->shape().size() == 2)
    {
        std::vector<std::vector<float>> result;
        for (std::size_t i = 0; i < other->shape()[0]; i++)
        {
            std::vector<float> result_i;
            for (std::size_t j = 0; j < other->shape()[1]; j++)
            {
                result_i.push_back(item() + (*other)(i, j));
            }
            result.push_back(result_i);
        }
        if (_requires_grad || other->requires_grad())
        {

            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                float grad_self = 0.0f;
                for (std::size_t i = 0; i < grad_output.size(); i++)
                {
                    grad_self += grad_output[i];
                }
                self->add_to_grad({grad_output[0]});
                other->add_to_grad({grad_output[0]});
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }

    // 1D + scalar

    if (_shape.size() == 1 && other->shape().size() == 0)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < shape()[0]; i++)
        {
            result.push_back(operator()(i) + other->item());
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D + scalar

    if (_shape.size() == 2 && other->shape().size() == 0)
    {
        std::vector<std::vector<float>> result;
        for (std::size_t i = 0; i < other->shape()[0]; i++)
        {
            std::vector<float> result_i;
            for (std::size_t j = 0; j < other->shape()[1]; j++)
            {
                result_i.push_back(item() + (*other)(i, j));
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }

    // 1D + 1D

    if (_shape[0] != other->shape().size())
    {
        throw std::invalid_argument("Dimention is not equal");
    }

    if (_shape.size() == 1)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < shape()[0]; i++)
        {
            result.push_back(operator()(i) + (*other)(i));
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                self->add_to_grad({grad_output});
                other->add_to_grad({grad_output});
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D + 2D
    else
    {
        if (shape()[1] != other->shape()[1])
        {
            throw std::invalid_argument("Second Dimention is not equal");
        }

        std::vector<std::vector<float>> result;
        for (std::size_t i = 0; i < shape()[0]; i++)
        {
            std::vector<float> result_i;
            for (std::size_t j = 0; j < shape()[1]; j++)
            {
                result_i.push_back(operator()(i, j) + (*other)(i, j));
            }
            result.push_back(result_i);
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                self->add_to_grad({grad_output});
                other->add_to_grad({grad_output});
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other)
{
    if (_shape.size() == 0 || other->shape().size() == 0)
    {
        throw std::invalid_argument("Both need to be at least one dimension");
    }

    // 1D x 1D -> float
    if (_shape.size() == 1 && other->shape().size() == 1)
    {
        float result = 0.0f;
        for (std::size_t i = 0; i < _shape[0]; i++)
        {
            result += operator()(i) * (*other)(i);
        }

        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                std::vector<float> grad_self;
                std::vector<float> grad_other;
                for (std::size_t i = 0; i < self->numel(); i++)
                {
                    grad_self.push_back((*other)(i)*grad_output[0]);
                    grad_other.push_back((*other)(i)*grad_output[0]);
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D x 1D -> 1D
    else if (_shape.size() == 2 && other->shape().size() == 1)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < _shape[0]; i++)
        {
            float result_i = 0;
            for (std::size_t j = 0; j < _shape[0]; j++)
            {
                result_i += operator()(i, j) * (*other)(i, j);
            }
            result.push_back(result_i);
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                std::vector<float> grad_self;
                for (std::size_t i = 0; i < self->shape()[0]; i++)
                {
                    for (std::size_t j = 0; j < self->shape()[1]; j++)
                    {
                        grad_self.push_back((*other)(j)*grad_output[i]);
                    }
                }
                std::vector<float> grad_other;
                for (std::size_t i = 0; i < self->shape()[0]; i++)
                {
                    float grad_other_i = 0.0f;
                    for (std::size_t j = 0; j < self->shape()[0]; j++)
                    {
                        grad_other_i += (*self)(j, i) * grad_output[j];
                    }
                    grad_other.push_back(grad_other_i);
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_self);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }

    // 1D x 2D -> 1D
    else if (_shape.size() == 1 && other->shape().size() == 2)
    {
        std::vector<float> result;
        for (std::size_t i = 0; i < other->shape()[1]; i++)
        {
            float result_i = 0;
            for (std::size_t j = 0; j < other->shape()[0]; j++)
            {
                result_i += operator()(j) * (*other)(j, i);
            }
            result.push_back(result_i);
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D x 2D -> 2D
    else
    {
        if (other->shape().size() < 2)
        {
            throw std::invalid_argument("Expected second tensor to have at least 2 dimensions for this operation");
        }

        std::vector<std::vector<float>> result;
        for (std::size_t i = 0; i < shape()[0]; i++)
        {
            std::vector<float> result_i;
            for (std::size_t j = 0; j < other->shape()[1]; j++)
            {
                float result_i_j = 0.0f;
                for (std::size_t k = 0; k < shape()[1]; k++)
                {
                    // 核心矩阵乘法公式：C[i][j] = sum(A[i][k] * B[k][j])
                    result_i_j += (*this)(i, k) * (*other)(k, j);
                }
                result_i.push_back(result_i_j);
            }
            result.push_back(result_i);
        }
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn =
                [self, other](const std::vector<float> &grad_output)
            {
                std::vector<float> grad_self;
                for (std::size_t i = 0; i < self->shape()[0]; i++)
                {
                    for (std::size_t j = 0; j < self->shape()[1]; j++)
                    {
                        float grad_self_i_j = 0.0f;
                        for (std::size_t k = 0; k < other->shape()[1]; k++)
                        {
                            grad_self_i_j +=
                                (*other)(j, k) * grad_output[i * other->shape()[1] + k];
                        }
                        grad_self.push_back(grad_self_i_j);
                    }
                }
                std::vector<float> grad_other;
                for (std::size_t i = 0; i < self->shape()[0]; i++)
                {
                    for (std::size_t j = 0; j < self->shape()[1]; j++)
                    {
                        float grad_other_i_j = 0.0f;
                        for (std::size_t k = 0; k < other->shape()[0]; k++)
                        {
                            grad_other_i_j +=
                                (*other)(k, i) * grad_output[k * other->shape()[1] + j];
                        }
                        grad_self.push_back(grad_other_i_j);
                    }
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_self);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
}

const bool &Tensor::requires_grad() const { return _requires_grad; };

const std ::vector<float> &Tensor::grad() const { return _grad; };

void Tensor::add_to_grad(const std::vector<float> &grad_upgrade)
{

    if (!requires_grad())
    {
        return;
    }

    if (_grad.size() != grad_upgrade.size())
    {
        throw std::runtime_error("Gradient shape dismath ");
    }
    for (std::size_t i = 0; i < _grad.size(); i++)
    {
        _grad[i] = grad_upgrade[i];
    }
}

void Tensor::zero_grad() { _grad = std::vector<float>(_data.size(), 0.0f); }

std::size_t Tensor::numel() const { return _data.size(); }
