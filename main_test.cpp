#include <iostream>

#include "PSO.h"
#include "FitnessFunction.h"

#include <time.h>				// time()

int main()
{
	// 粒子群优化器参数：2为变量维度，true表示有搜索上下限
	PSOPara psopara(2, true);
	psopara.particle_num_ = 20;		// 粒子个数
	psopara.max_iter_num_ = 300;	// 最大迭代次数
	psopara.dt_[0] = 1.0;			// 第一维度上的时间步长
	psopara.dt_[1] = 1.0;			// 第二维度上的时间步长
	psopara.wstart_[0] = 0.9;		// 第一维度上的起始权重系数
	psopara.wstart_[1] = 0.9;		// 第二维度上的起始权重系数
	psopara.wend_[0] = 0.4;			// 第一维度上的终止权重系数
	psopara.wend_[1] = 0.4;			// 第二维度上的终止权重系数
	psopara.C1_[0] = 1.49445;		// 第一维度上的加速度因子
	psopara.C1_[1] = 1.49445;
	psopara.C2_[0] = 1.49445;		// 第二维度上的加速度因子
	psopara.C2_[1] = 1.49445;

	// 如果有搜索上下限，则设置上下限
	psopara.lower_bound_[0] = -1.0;	// 第一维度搜索下限
	psopara.lower_bound_[1] = -1.0;	// 第二维度搜索下限
	psopara.upper_bound_[0] = 1.0;	// 第一维度搜索上限
	psopara.upper_bound_[1] = 1.0;	// 第二维度搜索上限

	
	PSOOptimizer psooptimizer(&psopara, FitnessFunction);

	std::srand((unsigned int)time(0));
	psooptimizer.InitialAllParticles();
	double fitness = psooptimizer.all_best_fitness_;
	double *result = new double[psooptimizer.max_iter_num_];

	for (int i = 0; i < psooptimizer.max_iter_num_; i++)
	{
		psooptimizer.UpdateAllParticles();
		result[i] = psooptimizer.all_best_fitness_;
		std::cout << "第" << i << "次迭代结果：";
		std::cout << "x = " << psooptimizer.all_best_position_[0] << ", " << "y = " << psooptimizer.all_best_position_[1];
		std::cout << ", fitness = " << result[i] << std::endl;
	}

	system("pause");
}