def my_lin_fit(x, y):
    n = len(x)
    
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    numerator_a = sum((x_i - x_mean) * (y_i - y_mean) for x_i, y_i in zip(x, y))
    
    denominator_a = sum((x_i - x_mean)**2 for x_i in x)

    a = numerator_a / denominator_a
        
    b = y_mean - a * x_mean
        
    return a, b


# def my_lin_fit(x, y):
#     n = len(x)
#     x_mean = np.mean(x)
#     y_mean = np.mean(y)
#     numerator_a = np.sum(x * y) - n * x_mean * y_mean
#     denominator_a = np.sum(x**2) - n * x_mean**2
#     a = numerator_a / denominator_a
#     b = y_mean - a * x_mean
#     return a, b

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.random.uniform(-2, 5, 10)
    y = np.random.uniform(0, 3, 10)
    

    a, b = my_lin_fit(x, y) 
    
    plt.plot(x, y, 'kx')
    
    xp = np.arange(-2, 5, 0.1)
    plt.plot(xp, a * xp + b, 'r-')
    
    print(f"My fit: a={a} and b={b}")
    plt.show()

