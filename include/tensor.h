#pragma once
#include <vector>
#include<iostream>
class Tensor
{
private:
    std::vector<float> _data;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _stride;

public:
    Tensor(float data);
    Tensor(std::vector<float> data);
    Tensor(std::vector<std::vector<float>> data);
    const std::vector<std::size_t>& shape() const { return _shape; }
    const std::vector<std::size_t>& stride() const { return _stride; }
    const float &item() const;
    float &item();
    const float &operator()(std::size_t i) const;
    float &operator()(std::size_t i);
    const float &operator()(std::size_t i, std::size_t j) const;
    float &operator()(std::size_t i, std::size_t j);

    friend std::ostream &operator<<(std::ostream &os , const Tensor &obj);
    std::shared_ptr<Tensor>operator+(std::shared_ptr<Tensor>other);

};