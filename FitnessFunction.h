#pragma once
#include "PSO.h"
#include <math.h>
#define PI 3.1415926

double FitnessFunction(Particle& particle);

double FitnessFunction(Particle& particle)
{
	double x = particle.position_[0];
	double y = particle.position_[1];
	double temp = sqrt(x * x + y * y);
	double fitness = sin(temp) / temp + exp(0.5* cos(2 * PI * x) + 0.5  * cos(2 * PI * y)) - 2.71289;
	return fitness;
}