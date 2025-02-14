# -*- coding: utf-8 -*-
"""
    Secret Sharing
    ~~~~~

    :copyright: (c) 2014 by Halfmoon Labs
    :license: MIT, see LICENSE for more details.
"""

from utilitybelt import secure_randint as randint


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def mod_inverse(k, prime):
    k = k % prime
    if k < 0:
        r = egcd(prime, -k)[2]
    else:
        r = egcd(prime, k)[2]
    return (prime + r) % prime


def random_polynomial(degree, intercept, upper_bound):
    """ Generates a random polynomial with positive coefficients.
    """
    if degree < 0:
        raise ValueError('Degree must be a non-negative number.')
    coefficients = [intercept]
    for i in range(degree):
        random_coeff = randint(0, upper_bound - 1)
        coefficients.append(random_coeff)
    return coefficients


def get_polynomial_points(coefficients, num_points, prime):
    """ Calculates the first n polynomial points.
        [ (1, f(1)), (2, f(2)), ... (n, f(n)) ]
    """
    points = []
    for x in range(1, num_points + 1):
        # start with x=1 and calculate the value of y
        y = coefficients[0]
        # calculate each term and add it to y, using modular math
        for i in range(1, len(coefficients)):
            exponentiation = (x ** i) % prime
            term = (coefficients[i] * exponentiation) % prime
            y = (y + term) % prime
        # add the point to the list of points
        points.append((x, y))
    return points


def modular_lagrange_interpolation(x, points, prime, isecc):
    # break the points up into lists of x and y values
    x_values, y_values = zip(*points)
    # print("inside ", y_values[0].x)
    # initialize f(x) and begin the calculation: f(x) = SUM( y_i * l_i(x) )
    if isecc != 1 and isecc != 0:
        raise ValueError("isecc must be 0 or 1.")

    lagrange_coefficients = []

    numerator, denominator = 1, 1
    for j in range(len(points)):
        # don't compute a polynomial fraction if i equals j
        if 0 == j:
            continue
        # compute a fraction & update the existing numerator + denominator
        numerator = (numerator * (x - x_values[j])) % prime
        denominator = (denominator * (x_values[0] - x_values[j])) % prime
    # get the polynomial from the numerator + denominator mod inverse
    lagrange_polynomial = numerator * mod_inverse(denominator, prime)
    lagrange_coefficients.append(lagrange_polynomial % prime)

    if isecc == 0:
        f_x = (prime + (int(y_values[0]) * int(lagrange_polynomial))) % prime
    elif isecc == 1:
        f_x = y_values[0] * lagrange_polynomial

    for i in range(1, len(points)):
        # evaluate the lagrange basis polynomial l_i(x)
        numerator, denominator = 1, 1
        for j in range(len(points)):
            # don't compute a polynomial fraction if i equals j
            if i == j:
                continue
            # compute a fraction & update the existing numerator + denominator
            numerator = (numerator * (x - x_values[j])) % prime
            denominator = (denominator * (x_values[i] - x_values[j])) % prime
        # get the polynomial from the numerator + denominator mod inverse
        lagrange_polynomial = numerator * mod_inverse(denominator, prime)
        lagrange_coefficients.append(lagrange_polynomial % prime)
        # print("lagrange_polynomial ", i, " is ", lagrange_polynomial % prime)
        # print("y value [", i, "] is", y_values[i])
        # multiply the current y & the evaluated polynomial & add it to f(x)

        if isecc == 1:
            f_x = f_x + (y_values[i] * lagrange_polynomial)
        else:
            f_x = (prime + f_x + (int(y_values[i]) * int(lagrange_polynomial))) % prime
    return f_x, lagrange_coefficients
