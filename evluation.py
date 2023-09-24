import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns  # 添加Seaborn库

class ModelEvaluator:
    def __init__(self, data_file):
        self.data = pd.read_excel(data_file)    
    def calculate_rmse(self, actual_column, predicted_column):
        return np.sqrt(mean_squared_error(self.data[actual_column], self.data[predicted_column]))
    def calculate_correlation(self, actual_column, predicted_column):
        r, _ = pearsonr(self.data[actual_column], self.data[predicted_column])
        return r
    def calculate_rho(self, actual_column, predicted_column):
        rho, _ = spearmanr(self.data[actual_column], self.data[predicted_column])
        return rho    
    def generate_evaluation_report(self, output_file, columns_to_evaluate):
        evaluation_data = []
        for column in columns_to_evaluate:
            rmse = self.calculate_rmse(columns_to_evaluate[column],column,)
            correlation = self.calculate_correlation(columns_to_evaluate[column],column)
            rho = self.calculate_rho(columns_to_evaluate[column],column)
            evaluation_data.append({
                'Column_Name': column,
                'RMSE': rmse,
                'Correlation (r)': correlation,
                'Rank Correlation (ρ)': rho
            })
        evaluation_table = pd.DataFrame(evaluation_data)
        evaluation_table.to_csv(output_file, index=False)
        return evaluation_table
    def plot_violin_plot(self,grouped_columns, output):
        plt.figure(figsize=(12, 5))
        # 使用Seaborn绘制小提琴图
        df = pd.DataFrame(self.data)
        for act,pres in grouped_columns.items():
            list =[]
            list.append(act)
            for pre in pres:
                list.append(pre)
            # 绘制小提琴图
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=df[list], inner="points")
            plt.xlabel('')
            plt.ylabel('Fitness')
            plt.title(f"{act}")
            plt.tight_layout()
            output_png = f'{output}/{act}_violin_plot.png'
            plt.savefig(output_png)
            plt.show()   
if __name__ == "__main__":
    evaluator = ModelEvaluator('data/test.xlsx')
    output_file = 'output/evaluation_report.csv'
    
    # 列出表中要评估的列名,字典名：真实值，字典值：预测值
    columns_to_evaluate = {"Pred_SUM_ecnet":"Obs_SUM",
                           "Pred_5A_ecnet":"Obs_5A",
                           "Pred_SUM*5A_ecnet":"Obs_SUM*5A",
                           "Pred_SUM_mavenn":"Obs_SUM",
                           "Pred_5A_mavenn":"Obs_5A",
                           "Pred_SUM*5A_mavenn":"Obs_SUM*5A"}
    evaluator.generate_evaluation_report(output_file=output_file,columns_to_evaluate=columns_to_evaluate)#产生评估csv文件
    # 创建一个字典，将相同的真实值作为键，对应的预测值作为值
    grouped_columns = {}
    for pred_column, actual_column in columns_to_evaluate.items():
        if actual_column not in grouped_columns:
            grouped_columns[actual_column] = []
        grouped_columns[actual_column].append(pred_column)

    evaluator.plot_violin_plot(grouped_columns,output="png")  # 画小提琴图
    for actual_column, pred_columns in grouped_columns.items():
        plt.figure(figsize=(12, 5))
        plt.xlabel(f'Actual {actual_column}')
        plt.ylabel(f'Predicted Values')
        plt.title(f'Scatter Plot for {actual_column}')
        for pred_column in pred_columns:
            # 绘制散点图
            plt.scatter(evaluator.data[actual_column], evaluator.data[pred_column], alpha=0.7, label=pred_column)     
            # 添加线性回归线
            lr = LinearRegression()
            lr.fit(np.array(evaluator.data[actual_column]).reshape(-1, 1), evaluator.data[pred_column])
            plt.plot(evaluator.data[actual_column], lr.predict(np.array(evaluator.data[actual_column]).reshape(-1, 1)), alpha=0.7)
        # 添加对角参考线
        plt.plot(evaluator.data[actual_column], evaluator.data[actual_column], linestyle='--', color='gray', alpha=0.7)
        plt.legend(loc='upper left')
        plt.tight_layout()
        output_png = f'png/{actual_column}_scatter_plot.png'
        plt.savefig(output_png)
        plt.show()
    print("Scatter Plots generated.")


