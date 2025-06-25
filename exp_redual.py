from exp.exp_basic import Exp_Basic
from models import redual  # 修改：导入新的双分支模型
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

warnings.filterwarnings('ignore')


def generate_timestamp():
    """生成时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_detailed_timestamp():
    """生成详细的时间戳字符串"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Exp_Fused_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Fused_Forecast, self).__init__(args)
        self.experiment_timestamp = generate_timestamp()
        self.detailed_timestamp = generate_detailed_timestamp()
        print(f"实验时间戳: {self.experiment_timestamp}")
        print(f"详细时间戳: {self.detailed_timestamp}")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 模型前向
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                # 修复：在使用f_dim之前先定义它
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

                # 收集预测和真实值用于计算R²
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        total_loss = np.average(total_loss)

        # 计算验证集上的R²
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        _, _, _, _, _ = metric(preds, trues)
        from sklearn.metrics import r2_score
        r2 = r2_score(trues.flatten(), preds.flatten())

        self.model.train()
        return total_loss, r2

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 修改：添加时间戳到路径
        timestamped_setting = f"{setting}_{self.experiment_timestamp}"
        path = os.path.join(self.args.checkpoints, timestamped_setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        # 设置早停patience为5，最大epoch为25
        early_stopping = EarlyStopping(patience=5, verbose=True)
        max_epochs = 25

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=max_epochs,
            max_lr=self.args.learning_rate
        )

        # 记录训练历史
        train_history = {
            'train_loss': [],
            'vali_loss': [],
            'vali_r2': [],
            'test_loss': [],
            'test_r2': []
        }

        # 创建总的epoch进度条
        epoch_pbar = tqdm(range(max_epochs), desc="Training Epochs", unit="epoch")

        for epoch in epoch_pbar:
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            # 为每个epoch的batch创建进度条
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}",
                              leave=False, unit="batch")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(batch_pbar):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 前向传播
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # 更新：batch进度条显示当前loss
                batch_pbar.set_postfix({'Loss': f"{loss.item():.6f}"})

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((max_epochs - epoch) * train_steps - i)
                    tqdm.write(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    tqdm.write(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    scheduler.step()

            batch_pbar.close()

            epoch_cost_time = time.time() - epoch_time
            train_loss = np.average(train_loss)
            vali_loss, vali_r2 = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_r2 = self.vali(test_data, test_loader, criterion)

            # 记录历史
            train_history['train_loss'].append(train_loss)
            train_history['vali_loss'].append(vali_loss)
            train_history['vali_r2'].append(vali_r2)
            train_history['test_loss'].append(test_loss)
            train_history['test_r2'].append(test_r2)

            # 使用tqdm.write输出结果，避免与进度条冲突
            tqdm.write(f"Epoch: {epoch + 1} cost time: {epoch_cost_time:.2f}s")
            tqdm.write(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            tqdm.write(f"Vali R²: {vali_r2:.4f} Test R²: {test_r2:.4f}")

            # 更新：epoch进度条显示当前指标
            epoch_pbar.set_postfix({
                'Train_Loss': f"{train_loss:.6f}",
                'Vali_R2': f"{vali_r2:.4f}",
                'Test_R2': f"{test_r2:.4f}"
            })

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                tqdm.write("Early stopping triggered!")
                tqdm.write(f"Training stopped at epoch {epoch + 1}")
                break

        epoch_pbar.close()

        # 保存训练历史
        history_file = os.path.join(path, f'training_history_{self.experiment_timestamp}.json')
        train_history_serializable = {}
        for key, value in train_history.items():
            train_history_serializable[key] = [float(x) for x in value]

        import json
        with open(history_file, 'w') as f:
            json.dump(train_history_serializable, f, indent=2)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 最终训练完成后计算训练集R²
        tqdm.write("Computing final training metrics...")
        final_train_loss, final_train_r2 = self.train_step_metrics(train_loader, criterion)
        tqdm.write(f"Final Training R²: {final_train_r2:.6f}")

    def train_step_metrics(self, train_loader, criterion):
        """计算训练集上的R²指标（可选调用）"""
        train_preds = []
        train_trues = []
        train_loss = []

        self.model.eval()
        with torch.no_grad():
            # 添加：为训练集指标计算添加进度条
            metric_pbar = tqdm(enumerate(train_loader), desc="Computing train metrics",
                               total=min(100, len(train_loader)), leave=False)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in metric_pbar:
                if i >= 100:  # 只计算前100个batch，避免计算时间过长
                    break

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                train_preds.append(outputs.detach().cpu().numpy())
                train_trues.append(batch_y.detach().cpu().numpy())

            metric_pbar.close()

        if train_preds:
            train_preds = np.concatenate(train_preds, axis=0)
            train_trues = np.concatenate(train_trues, axis=0)
            # 修复：使用与vali函数相同的R²计算方式
            _, _, _, _, _ = metric(train_preds, train_trues)
            from sklearn.metrics import r2_score
            train_r2 = r2_score(train_trues.flatten(), train_preds.flatten())
        else:
            train_r2 = 0.0

        self.model.train()
        return np.average(train_loss), train_r2

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        # 修改：添加时间戳到设置
        timestamped_setting = f"{setting}_{self.experiment_timestamp}"

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + timestamped_setting, 'checkpoint.pth')))

        preds = []
        trues = []

        # 修改：创建带时间戳的测试结果文件夹
        folder_path = './test_results/' + timestamped_setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="Testing", unit="batch")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_pbar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

            test_pbar.close()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        from sklearn.metrics import r2_score
        r2 = r2_score(trues.flatten(), preds.flatten())
        print(f'mse:{mse:.6f}, mae:{mae:.6f}')
        print(f'rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}')
        print(f'R²:{r2:.6f}')

        # 修改：保存结果到带时间戳的文件夹
        folder_path = './results/' + timestamped_setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # === 反归一化逻辑保持不变 ===
        print("开始反归一化处理...")
        print(f"原始preds形状: {preds.shape}")
        print(f"原始trues形状: {trues.shape}")

        # 获取原始测试数据
        raw_df = test_data.raw_test_df
        print("原始数据列名:", raw_df.columns.tolist())

        # 计算预测数据的数量
        num_preds = len(preds.flatten())
        print(f"预测数据点数量: {num_preds}")

        # 方法1：如果features=='S'（单变量），直接反归一化
        if self.args.features == 'S':
            print("使用单变量反归一化方法")
            preds_unscaled = test_data.inverse_transform(preds.reshape(-1, 1)).flatten()
            trues_unscaled = test_data.inverse_transform(trues.reshape(-1, 1)).flatten()
        else:
            print("使用多变量反归一化方法")
            feature_dim = test_data.data_x.shape[-1]
            print(f"特征维度: {feature_dim}")

            if hasattr(test_data, 'scaler'):
                test_mean = np.mean(test_data.data_x, axis=0)

                preds_full = np.tile(test_mean, (num_preds, 1))
                preds_full[:, -1] = preds.flatten()

                trues_full = np.tile(test_mean, (num_preds, 1))
                trues_full[:, -1] = trues.flatten()

                preds_unscaled = test_data.inverse_transform(preds_full)[:, -1]
                trues_unscaled = test_data.inverse_transform(trues_full)[:, -1]
            else:
                print("警告：无法找到scaler，使用原始数据")
                preds_unscaled = preds.flatten()
                trues_unscaled = trues.flatten()

        print(f"反归一化后预测值范围: [{preds_unscaled.min():.6f}, {preds_unscaled.max():.6f}]")
        print(f"反归一化后真实值范围: [{trues_unscaled.min():.6f}, {trues_unscaled.max():.6f}]")

        # 获取对应的cycle数据
        start_idx = self.args.seq_len
        end_idx = start_idx + num_preds

        print(f"从原始数据获取cycle，索引范围: {start_idx} 到 {end_idx}")
        print(f"原始数据长度: {len(raw_df)}")

        # 检查数据集中是否有Cycle列
        cycle_col = None
        date_col = None

        possible_cycle_names = ['Cycle', 'cycle', 'CYCLE', 'cycle_number', 'Cycle_Number']
        for col in possible_cycle_names:
            if col in raw_df.columns:
                cycle_col = col
                break

        possible_date_names = ['date', 'Date', 'DATE', 'time', 'Time', 'timestamp']
        for col in possible_date_names:
            if col in raw_df.columns:
                date_col = col
                break

        print(f"找到的Cycle列: {cycle_col}")
        print(f"找到的Date列: {date_col}")

        if end_idx <= len(raw_df):
            if cycle_col is not None:
                cycle_data = raw_df[cycle_col].values[start_idx:end_idx]
                print("使用数据集中的Cycle列")
            else:
                cycle_data = np.arange(start_idx + 1, end_idx + 1)
                print("数据集中没有Cycle列，使用生成的序号")

            if date_col is not None:
                date_data = raw_df[date_col].values[start_idx:end_idx]
            else:
                date_data = None

            true_targets = raw_df[self.args.target].values[start_idx:end_idx]
        else:
            print("警告：索引超出范围，调整索引")
            if cycle_col is not None:
                cycle_data = raw_df[cycle_col].values[-num_preds:]
            else:
                cycle_data = np.arange(len(raw_df) - num_preds + 1, len(raw_df) + 1)

            if date_col is not None:
                date_data = raw_df[date_col].values[-num_preds:]
            else:
                date_data = None

            true_targets = raw_df[self.args.target].values[-num_preds:]

        print(f"获取的cycle数据长度: {len(cycle_data)}")
        print(f"cycle数据类型: {type(cycle_data[0])}")
        print(f"cycle数据前5个: {cycle_data[:5]}")
        print(f"获取的真实target长度: {len(true_targets)}")
        print(f"真实target范围: [{true_targets.min():.6f}, {true_targets.max():.6f}]")

        min_length = min(len(cycle_data), len(true_targets), len(preds_unscaled))
        print(f"最终使用的数据长度: {min_length}")

        # 修改：创建结果DataFrame时添加实验信息
        results_df = pd.DataFrame({
            'Cycle': cycle_data[:min_length],
            'True_Target': true_targets[:min_length],
            'Predicted_Target': preds_unscaled[:min_length]
        })

        # 添加实验元信息
        experiment_info = pd.DataFrame({
            'Experiment_Timestamp': [self.detailed_timestamp] * min_length,
            'Model_Name': [self.args.model] * min_length,
            'Dataset': [self.args.data] * min_length,
            'Data_Path': [self.args.data_path] * min_length,
            'MSE': [mse] * min_length,
            'MAE': [mae] * min_length,
            'RMSE': [rmse] * min_length,
            'R2': [r2] * min_length,
            'Final_d_model': [self.args.d_model] * min_length,
            'Final_learning_rate': [self.args.learning_rate] * min_length,
            'Final_dropout': [self.args.dropout] * min_length,
        })

        # 合并实验信息和结果
        detailed_results_df = pd.concat([experiment_info, results_df], axis=1)

        # 保存详细结果CSV文件
        results_csv_path = os.path.join(folder_path, f'forecast_results_{self.experiment_timestamp}.csv')
        detailed_results_df.to_csv(results_csv_path, index=False)
        print(f"详细结果已保存至 {results_csv_path}")

        # 同时保存简化版本（保持原有格式兼容性）
        simple_results_csv_path = os.path.join(folder_path, 'forecast_results.csv')
        results_df.to_csv(simple_results_csv_path, index=False)
        print(f"简化结果已保存至 {simple_results_csv_path}")

        print("前5行结果预览:")
        print(results_df.head())

        print("开始生成可视化图表...")

        # 设置matplotlib的科技论文风格
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        # 确定x轴数据：优先使用Cycle，如果用户明确需要date则使用date
        # 这里我们使用Cycle作为x轴，因为这是电池研究的标准做法
        x_data = cycle_data[:min_length]
        x_label = 'Cycle'

        print(f"使用x轴数据: {x_label}")
        print(f"x轴数据范围: {x_data.min()} 到 {x_data.max()}")

        # 绘制真实值（蓝线）
        ax.plot(x_data, true_targets[:min_length],
                color='#2E86AB', linewidth=2.5, alpha=0.8,
                label='True SoH', marker='o', markersize=3, markevery=max(1, min_length // 50))

        # 绘制预测值（红线）
        ax.plot(x_data, preds_unscaled[:min_length],
                color='#F24236', linewidth=2.5, alpha=0.8,
                label='Predicted SoH', marker='s', markersize=3, markevery=max(1, min_length // 50))

        # 设置标题和标签
        ax.set_title('Battery SoH Prediction - Dual Branch Model', fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel('State of Health (SoH)', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=14, fontweight='bold')

        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # 添加图例（右上角）
        legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                           fancybox=True, shadow=True, framealpha=0.9,
                           bbox_to_anchor=(0.98, 0.98))
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(0.8)

        # 添加性能指标文本框（右上角，图例下方）
        textstr = f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}'
        props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8, edgecolor='gray')
        ax.text(0.98, 0.75, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=props, family='monospace')

        # 设置轴的范围和刻度
        y_min, y_max = min(np.min(true_targets[:min_length]), np.min(preds_unscaled[:min_length])), \
            max(np.max(true_targets[:min_length]), np.max(preds_unscaled[:min_length]))
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

        # 美化刻度
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=6)
        ax.tick_params(axis='both', which='minor', width=0.8, length=3)

        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('gray')

        # 添加阴影区域显示预测误差
        error = np.abs(true_targets[:min_length] - preds_unscaled[:min_length])
        ax.fill_between(x_data,
                        preds_unscaled[:min_length] - error / 2,
                        preds_unscaled[:min_length] + error / 2,
                        alpha=0.2, color='red', label='Prediction Error Band')

        # 调整布局
        plt.tight_layout()

        # 保存高质量图片
        plot_path = os.path.join(folder_path, 'dual_branch_battery_soh_prediction.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        # 同时保存PDF格式（适合论文使用）
        pdf_path = os.path.join(folder_path, 'dual_branch_battery_soh_prediction.pdf')
        plt.savefig(pdf_path, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        print(f"可视化图表已保存:")
        print(f"PNG格式: {plot_path}")
        print(f"PDF格式: {pdf_path}")

        # 显示图表（可选，如果在jupyter notebook中运行）
        plt.show()
        plt.close()

        # =================== 可视化功能结束 ===================

        # 保存指标
        np.save(folder_path + f'metrics_{self.experiment_timestamp}.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + f'pred_{self.experiment_timestamp}.npy', preds)
        np.save(folder_path + f'true_{self.experiment_timestamp}.npy', trues)

        # 修改：保存到带时间戳的文本文件
        result_file = f"result_dual_branch_forecast_{self.experiment_timestamp}.txt"
        f = open(result_file, 'a', encoding='utf-8')
        f.write(f"Experiment Time: {self.detailed_timestamp}\n")
        f.write(f"Dataset: {self.args.data_path}\n")
        f.write(f"Model: {self.args.model}\n")
        f.write(timestamped_setting + "\n")
        f.write(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}, R2:{r2:.6f}\n')
        f.write('\n')
        f.close()

        # 同时保存到总的结果文件（保持原有逻辑）
        f = open("result_dual_branch_forecast.txt", 'a', encoding='utf-8')
        f.write(f"[{self.detailed_timestamp}] " + timestamped_setting + "\n")
        f.write(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}, R2:{r2:.6f}\n')
        f.write('\n')
        f.close()

        return