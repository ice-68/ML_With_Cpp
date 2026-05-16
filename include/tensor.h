#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <functional>


//私有成员全带下划线_，传入参数没下划线。这是规范。

class Tensor : public std::enable_shared_from_this<Tensor>
{
private:
    std::vector<float> _data;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _stride;
    std::vector<float> _grad;
    std::function<void(const std::vector<float> &)> _gradfn;
    std::vector<std::shared_ptr<Tensor>> _parents;
    bool _requires_grad;
    void _backward();
    bool _visited = false;
    void _reset_graph_visit();

public:
    Tensor(float data, bool requires_grad = false,
           std::function<void(const std::vector<float> &)> gradfn = nullptr,
           std::vector<std::shared_ptr<Tensor>> parents = {});

    Tensor(std::vector<float> data, bool requires_grad = false,
           std::function<void(const std::vector<float> &)> gradfn = nullptr,
           std::vector<std::shared_ptr<Tensor>> parents = {});

    Tensor(std::vector<std::vector<float>> data, bool requires_grad = false,
           std::function<void(const std::vector<float> &)> gradfn = nullptr,
           std::vector<std::shared_ptr<Tensor>> parents = {});

    const std::vector<std::size_t> &shape() const { return _shape; }
    const std::vector<std::size_t> &stride() const { return _stride; }
    const float &item() const;
    float &item();
    const float &operator()(std::size_t i) const;
    float &operator()(std::size_t i);
    const float &operator()(std::size_t i, std::size_t j) const;
    float &operator()(std::size_t i, std::size_t j);

    friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);

    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);

    const bool &requires_grad() const;
    const std::vector<float> &grad() const;
    std::vector<float> &grad();
    void backward();
    void add_to_grad(const std::vector<float> &grad_update);
    void zero_grad();
    std::size_t numel() const;
    std::vector<float> &data();
};