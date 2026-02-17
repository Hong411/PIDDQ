# -*- coding: utf-8 -*-
import utils
from configs import set_plot_style, ERROR_IDS
from collections import defaultdict
import numpy as np
from data.data_processing import DatasetContainer
from models import gpr_operations, nn_operations
import matplotlib.pyplot as plt
from utils import GPUCalculator
from sklearn.metrics import roc_curve, auc
from data import external_data, data_loader

def prepare_k_fold_vali_dataset(data, model_name, vali_dict_name, model='gpr'):
    vali_dict ={}
    vali_dict_train= {}
    
    k_fold_id_dict = utils.load_dict(vali_dict_name)
    
    for key in k_fold_id_dict:
        # key = fold number, type = str
        i = int(key)
        
        vali_dict[key] = defaultdict(list)
        vali_dict_train[key] = defaultdict(list)
        
        vali_model = utils.load_model(f'{model_name}_{i}')

        vali_dict[key] = defaultdict(list)
        
        for i, id_val in enumerate(data.ids):
            
            if id_val in k_fold_id_dict[key]['val_id']:
                vali_dict[key]['id_list'].append(id_val)
                vali_dict[key]['sim_list'].append(data.features[i])
                vali_dict[key]['exp_list'].append(data.targets[i])
                vali_dict[key]['mw_int_list'].append(data.mws[i])
                vali_dict[key]['smiles_list'].append(data.smiles[i])
                
            elif id_val in k_fold_id_dict[key]['train_id']:
                vali_dict_train[key]['id_list'].append(id_val)
                vali_dict_train[key]['sim_list'].append(data.features[i])
                vali_dict_train[key]['exp_list'].append(data.targets[i])
                vali_dict_train[key]['mw_int_list'].append(data.mws[i])
                vali_dict_train[key]['smiles_list'].append(data.smiles[i])
        if model=='gpr':                  
            vali_pred = gpr_operations.predict_tf(vali_model, vali_dict[key]['sim_list'])
            vali_dict[key]['pred_list'] = vali_pred
            
            vali_train_pred = gpr_operations.predict_tf(vali_model, vali_dict_train[key]['sim_list'])
            vali_dict_train[key]['pred_list'] = vali_train_pred
            
        elif model=='nn':
            
            vali_pred = nn_operations.predict(vali_model, vali_dict[key]['sim_list'])
            vali_dict[key]['pred_list'] = vali_pred
            
            vali_train_pred = nn_operations.predict(vali_model, vali_dict_train[key]['sim_list'])
            vali_dict_train[key]['pred_list'] = vali_train_pred
            
    return vali_dict, vali_dict_train 

def data_prepare():
    db_id_pro, db_sim_pro, db_smiles_pro, db_mw_pro = utils.load_dict('db_pro_info')
    
    # train and test data for dft and exp
    train_data = utils.load_dict('unique_train_data_no_Dsub')
    test_data = utils.load_dict('unique_test_data_no_Dsub')
    cleaned_train_data = utils.load_dict('unique_clean_train_data_no_error_and_iminol')
    cleaned_test_data = utils.load_dict('unique_clean_test_data_no_error_and_iminol')
    
    # GPR model and dict
    modelname_uggnd_k5 = 'unique_ggnd_train0.5_k5'
    modelname_uggnder_k5 = 'unique_ggnder_train0.5_k5'
    modelname_uggnd_all = 'unique_ggnd_train_all'
    modelname_uggnder_all = 'unique_ggnder_train_all'
    
    dictname_vali_uggnd = 'unique_ggnd_5fold_dict'
    dictname_vali_uggnder = 'unique_ggnder_5fold_dict'
    
    # GPR vali and test pred
    vali_dict, vali_dict_train = prepare_k_fold_vali_dataset(train_data, modelname_uggnd_k5, dictname_vali_uggnd)
    vali_dict_er, vali_dict_train_er = prepare_k_fold_vali_dataset(cleaned_train_data, modelname_uggnder_k5, dictname_vali_uggnder)
    
    test_pred_ggnd = gpr_operations.predict_tf(utils.load_model(modelname_uggnd_all), test_data.features)
    test_pred_ggnder = gpr_operations.predict_tf(utils.load_model(modelname_uggnder_all), cleaned_test_data.features)
    
    utils.save_dict(vali_dict, 'unique_ggnd_vali_dict')
    utils.save_dict(vali_dict_train, 'unique_ggnd_vali_dict_train')
    utils.save_dict(vali_dict_er, 'unique_ggnder_vali_dict')
    utils.save_dict(vali_dict_train_er, 'unique_ggnder_vali_dict_train')
    utils.save_dict(test_pred_ggnd, 'unique_ggnd_test_pred')
    utils.save_dict(test_pred_ggnder, 'unique_ggnder_test_pred')
    
    test_pred_db = gpr_operations.predict_tf(utils.load_model(modelname_uggnd_all), db_sim_pro)
    test_pred_db_er = gpr_operations.predict_tf(utils.load_model(modelname_uggnder_all), db_sim_pro)
    
    utils.save_dict(test_pred_db, 'db_pro_pred')
    utils.save_dict(test_pred_db_er, 'db_pro_pred_er')
    
    vali_plot_dict = defaultdict(list)
    
    for key in vali_dict.keys():
        vali_plot_dict['id_list'].extend(vali_dict[key]['id_list'])
        vali_plot_dict['sim_list'].extend(vali_dict[key]['sim_list'])
        vali_plot_dict['exp_list'].extend(vali_dict[key]['exp_list'])
        vali_plot_dict['mw_int_list'].extend(vali_dict[key]['mw_int_list'])
        vali_plot_dict['smiles_list'].extend(vali_dict[key]['smiles_list'])
        vali_plot_dict['pred_list'].extend(vali_dict[key]['pred_list'])
    
    nist_plot_dict = {}    
    
    nist_plot_dict['id_list'] = np.concatenate((vali_plot_dict['id_list'], test_data.ids))
    nist_plot_dict['sim_list'] = np.concatenate((vali_plot_dict['sim_list'], test_data.features))
    nist_plot_dict['exp_list'] = np.concatenate((vali_plot_dict['exp_list'], test_data.targets))
    nist_plot_dict['mw_list'] = np.concatenate((vali_plot_dict['mw_int_list'], test_data.mws))
    nist_plot_dict['smiles_list'] = np.concatenate((vali_plot_dict['smiles_list'], test_data.smiles))
    nist_plot_dict['pred_list'] = np.concatenate((vali_plot_dict['pred_list'], test_pred_ggnd))
    
    vali_plot_dict_er = defaultdict(list)
    
    for key in vali_dict_er.keys():
        vali_plot_dict_er['id_list'].extend(vali_dict_er[key]['id_list'])
        vali_plot_dict_er['sim_list'].extend(vali_dict_er[key]['sim_list'])
        vali_plot_dict_er['exp_list'].extend(vali_dict_er[key]['exp_list'])
        vali_plot_dict_er['mw_int_list'].extend(vali_dict_er[key]['mw_int_list'])
        vali_plot_dict_er['smiles_list'].extend(vali_dict_er[key]['smiles_list'])
        vali_plot_dict_er['pred_list'].extend(vali_dict_er[key]['pred_list'])
    
    nist_plot_dict_er = {}    
    
    nist_plot_dict_er['id_list'] = np.concatenate((vali_plot_dict_er['id_list'], cleaned_test_data.ids))
    nist_plot_dict_er['sim_list'] = np.concatenate((vali_plot_dict_er['sim_list'], cleaned_test_data.features))
    nist_plot_dict_er['exp_list'] = np.concatenate((vali_plot_dict_er['exp_list'], cleaned_test_data.targets))
    nist_plot_dict_er['mw_list'] = np.concatenate((vali_plot_dict_er['mw_int_list'], cleaned_test_data.mws))
    nist_plot_dict_er['smiles_list'] = np.concatenate((vali_plot_dict_er['smiles_list'], cleaned_test_data.smiles))
    nist_plot_dict_er['pred_list'] = np.concatenate((vali_plot_dict_er['pred_list'], test_pred_ggnder))
    
    utils.save_dict(vali_plot_dict, 'unique_vali_plot_dict')
    utils.save_dict(vali_plot_dict_er, 'unique_vali_plot_dict_er')
    utils.save_dict(nist_plot_dict, 'unique_nist_plot_dict')
    utils.save_dict(nist_plot_dict_er, 'unique_nist_plot_dict_er')

    # NN model and dict
    modelname_unnnd_k5 = 'unique_nnnd_train0.5_k5'
    modelname_unnnder_k5 = 'unique_nnnder_train0.5_k5'
    modelname_unnnd_all = 'unique_nnnd_train_all'
    modelname_unnnder_all = 'unique_nnnder_train_all'
    
    dictname_vali_unnnd = 'unique_nnnd_5fold_dict'
    dictname_vali_unnnder = 'unique_nnnder_5fold_dict'
    
    vali_dict_nn, vali_dict_train_nn = prepare_k_fold_vali_dataset(train_data, 
                                                                   modelname_unnnd_k5, dictname_vali_unnnd, model='nn')
    vali_dict_er_nn, vali_dict_train_er_nn = prepare_k_fold_vali_dataset(cleaned_train_data, 
                                                                         modelname_unnnder_k5, dictname_vali_unnnder, model='nn')
    
    test_pred_nnnd = nn_operations.predict(utils.load_model(modelname_unnnd_all), test_data.features)
    test_pred_nnnder = nn_operations.predict(utils.load_model(modelname_unnnder_all), cleaned_test_data.features)
    
    utils.save_dict(vali_dict_nn, 'unique_nnnd_vali_dict')
    utils.save_dict(vali_dict_train_nn, 'unique_nnnd_vali_dict_train')
    utils.save_dict(vali_dict_er_nn, 'unique_nnnder_vali_dict')
    utils.save_dict(vali_dict_train_er_nn, 'unique_nnnder_vali_dict_train')
    utils.save_dict(test_pred_nnnd, 'unique_nnnd_test_pred')
    utils.save_dict(test_pred_nnnder, 'unique_nnnder_test_pred')
    
    vali_plot_dict_nn = defaultdict(list)
    
    for key in vali_dict_nn.keys():
        vali_plot_dict_nn['id_list'].extend(vali_dict_nn[key]['id_list'])
        vali_plot_dict_nn['sim_list'].extend(vali_dict_nn[key]['sim_list'])
        vali_plot_dict_nn['exp_list'].extend(vali_dict_nn[key]['exp_list'])
        vali_plot_dict_nn['mw_int_list'].extend(vali_dict_nn[key]['mw_int_list'])
        vali_plot_dict_nn['smiles_list'].extend(vali_dict_nn[key]['smiles_list'])
        vali_plot_dict_nn['pred_list'].extend(vali_dict_nn[key]['pred_list'])
    
    nist_plot_dict_nn = {}    
    
    nist_plot_dict_nn['id_list'] = np.concatenate((vali_plot_dict_nn['id_list'], test_data.ids))
    nist_plot_dict_nn['sim_list'] = np.concatenate((vali_plot_dict_nn['sim_list'], test_data.features))
    nist_plot_dict_nn['exp_list'] = np.concatenate((vali_plot_dict_nn['exp_list'], test_data.targets))
    nist_plot_dict_nn['mw_list'] = np.concatenate((vali_plot_dict_nn['mw_int_list'], test_data.mws))
    nist_plot_dict_nn['smiles_list'] = np.concatenate((vali_plot_dict_nn['smiles_list'], test_data.smiles))
    nist_plot_dict_nn['pred_list'] = np.concatenate((vali_plot_dict_nn['pred_list'], test_pred_nnnd))
    
    vali_plot_dict_er_nn = defaultdict(list)
    
    for key in vali_dict_er_nn.keys():
        vali_plot_dict_er_nn['id_list'].extend(vali_dict_er_nn[key]['id_list'])
        vali_plot_dict_er_nn['sim_list'].extend(vali_dict_er_nn[key]['sim_list'])
        vali_plot_dict_er_nn['exp_list'].extend(vali_dict_er_nn[key]['exp_list'])
        vali_plot_dict_er_nn['mw_int_list'].extend(vali_dict_er_nn[key]['mw_int_list'])
        vali_plot_dict_er_nn['smiles_list'].extend(vali_dict_er_nn[key]['smiles_list'])
        vali_plot_dict_er_nn['pred_list'].extend(vali_dict_er_nn[key]['pred_list'])
    
    nist_plot_dict_er_nn = {}    
    
    nist_plot_dict_er_nn['id_list'] = np.concatenate((vali_plot_dict_er_nn['id_list'], cleaned_test_data.ids))
    nist_plot_dict_er_nn['sim_list'] = np.concatenate((vali_plot_dict_er_nn['sim_list'], cleaned_test_data.features))
    nist_plot_dict_er_nn['exp_list'] = np.concatenate((vali_plot_dict_er_nn['exp_list'], cleaned_test_data.targets))
    nist_plot_dict_er_nn['mw_list'] = np.concatenate((vali_plot_dict_er_nn['mw_int_list'], cleaned_test_data.mws))
    nist_plot_dict_er_nn['smiles_list'] = np.concatenate((vali_plot_dict_er_nn['smiles_list'], cleaned_test_data.smiles))
    nist_plot_dict_er_nn['pred_list'] = np.concatenate((vali_plot_dict_er_nn['pred_list'], test_pred_nnnder))
    
    utils.save_dict(vali_plot_dict_nn, 'unique_vali_plot_dict_nn')
    utils.save_dict(vali_plot_dict_er_nn, 'unique_vali_plot_dict_er_nn')
    utils.save_dict(nist_plot_dict_nn, 'unique_nist_plot_dict_nn')
    utils.save_dict(nist_plot_dict_er_nn, 'unique_nist_plot_dict_er_nn')  

def plot_pcc_and_roc_models():
    set_natcomsci_style()
    
    def save_figure_pdf(fig, filename, width_cm=8, height_cm=5):
        """
        按照Nature要求保存PDF
        
        参数:
            fig: matplotlib图形对象
            filename: 文件名（无需扩展名）
            width_cm: 宽度（厘米）
            height_cm: 高度（厘米）
        """
        # 厘米转英寸（1英寸=2.54厘米）
        width_inch = width_cm / 2.54
        height_inch = height_cm / 2.54
        
        # 设置图形尺寸
        fig.set_size_inches(width_inch, height_inch)
        
        # 保存为PDF（符合矢量图要求）
        save_path = f"{filename}.pdf"
        fig.savefig(save_path, 
                    format='pdf',
                    bbox_inches='tight',
                    pad_inches=0.05,
                    dpi=600,
                    transparent=False)
        print(f"已保存: {save_path}")
        plt.close(fig)  # 关闭图形释放内存
    
    # ============================================
    # 3. 修改后的绘图函数
    # ============================================
    def plot_hist(sl_list, xlabel='Pearson Correlation Coefficient', 
                  width_cm=8, height_cm=5, save_name="histogram"):
        """
        绘制直方图（多子图）
        """
        max_index = len(sl_list) - 1
        range_min = 0.0
        range_max = 1.0
        num_bins = 100
        
        # 创建图形 - 使用constrained_layout自动调整
        fig, axs = plt.subplots(len(sl_list), 1, sharex=True, 
                               figsize=(width_cm/2.54, height_cm/2.54),
                               constrained_layout=True)
        print(fig.get_size_inches())
        
        # 如果只有一个子图，确保axs是列表
        if len(sl_list) == 1:
            axs = [axs]
        
        color_set = ['#7ABBDB', '#84BA42', '#A51C36']
        
        for i, sl in enumerate(sl_list):
            print(f"数据集 {i+1} 均值: {np.mean(sl):.4f}")
            hist, bins = np.histogram(sl, bins=num_bins, range=(range_min, range_max))
            axs[i].hist(bins[:-1], bins=bins, weights=hist, 
                       color=color_set[i % len(color_set)], 
                       edgecolor='black', linewidth=0.2)  # 细边框
        
        # 设置坐标轴
        axs[max_index].set_xlim(range_min, range_max)
        axs[max_index].set_xlabel(xlabel, fontsize=6)  # 去掉粗体
        
        # 添加Y轴标签（居中）
        fig.supylabel('#Cases', x=0.06, fontsize=6)
        plt.tight_layout(rect=[0.0, 0, 1, 1])
        # 保存PDF
        save_figure_pdf(fig, save_name, width_cm, height_cm)
    
    def plot_violin(sl_list, xlabel='Methods', ylabel='Pearson Correlation Coefficient', 
                    labels=None, width_cm=8, height_cm=5, save_name="violin"):
        """
        绘制小提琴图
        """
        if labels is None:
            labels = [f'Method {i+1}' for i in range(len(sl_list))]
        
        range_min = 0.4
        range_max = 1.0
        color_set = ['#7ABBDB', '#84BA42', '#A51C36']
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(width_cm/2.54, height_cm/2.54),
                              constrained_layout=True)
        
        # 绘制小提琴图
        parts = ax.violinplot(sl_list, positions=range(len(sl_list)), 
                             showmeans=True, showmedians=False)  # 只显示均值
        
        # 设置颜色
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(color_set[i % len(color_set)])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        # 设置其他元素颜色和线宽
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(1)
        parts['cbars'].set_color('black')
        parts['cbars'].set_linewidth(0.5)
        parts['cmins'].set_color('black')
        parts['cmins'].set_linewidth(0.5)
        parts['cmaxes'].set_color('black')
        parts['cmaxes'].set_linewidth(0.5)
        
        # 设置坐标轴
        ax.set_ylim(range_min, range_max)
        ax.set_ylabel(ylabel, fontsize=6)
        ax.set_xlabel(xlabel, fontsize=6)
        ax.set_yticks(np.arange(0.4, 1.05, 0.1))
        
        # 设置横坐标标签
        ax.set_xticks(range(len(sl_list)))
        ax.set_xticklabels(labels, fontsize=6)
        
        # 美化（Nature风格：简洁）
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # 网格（细网格）
        ax.grid(axis='y', alpha=0.3, linewidth=0.3)
        
        # 打印均值
        for i, sl in enumerate(sl_list):
            print(f'{labels[i]} 均值: {np.mean(sl):.4f}')
        
        # 保存PDF
        save_figure_pdf(fig, save_name, width_cm, height_cm)
    
    def plot_roc_list(matrix_list, labels=None, width_cm=8, height_cm=5, 
                      save_name="roc_comparison"):
        """
        绘制多个ROC曲线比较图
        """
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(width_cm/2.54, height_cm/2.54),
                              constrained_layout=True)
        
        # 如果没有提供labels，则自动生成
        if labels is None:
            labels = [f'Model {i+1}' for i in range(len(matrix_list))]
        elif len(labels) != len(matrix_list):
            raise ValueError("labels的长度必须与matrix_list的长度相同")
        
        color_set = ['#A51C36', '#84BA42', '#7ABBDB', '#F68B1F', '#6A4C9C']
        
        for i, iden_matrix in enumerate(matrix_list):
            # 确保输入是numpy数组
            if not isinstance(iden_matrix, np.ndarray):
                iden_matrix = np.array(iden_matrix)
            
            # 计算ROC曲线
            predicted_scores = []
            true_labels = []
            for r_idx, c_idx in np.ndindex(iden_matrix.shape):
                true_labels.append(1 if r_idx == c_idx else 0)
                predicted_scores.append(iden_matrix[r_idx, c_idx])
            
            fpr, tpr, _ = roc_curve(true_labels, predicted_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            # 绘制曲线（按顺序取颜色）
            color = color_set[i % len(color_set)]
            ax.plot(fpr, tpr, color=color, lw=1.2,
                    label=f'{labels[i]} (AUC={roc_auc:.3f})')
        
        # 绘制对角线参考线
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
        
        # 设置图表属性
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=6)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=6)
        
        # 图例（位置优化）
        ax.legend(loc="lower right", fontsize=6, frameon=False)
        
        # 网格
        ax.grid(True, alpha=0.3, linewidth=0.3)
        
        # 坐标轴线宽
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            
        plt.tight_layout(rect=[0.0, 0, 1, 1])
        
        # 保存PDF
        save_figure_pdf(fig, save_name, width_cm, height_cm)
        
    def dft_data_plot():
        nist_plot_dict = utils.load_dict('unique_nist_plot_dict')
        nist_plot_dict_er = utils.load_dict('unique_nist_plot_dict_er')
        
        nist_plot_dict_nn = utils.load_dict('unique_nist_plot_dict_nn')
        nist_plot_dict_er_nn = utils.load_dict('unique_nist_plot_dict_er_nn')
        
        sl_dft = [utils.pearson(u, v) for u, v in zip(nist_plot_dict['sim_list'], nist_plot_dict['exp_list'])] 
        sl_gpr = [utils.pearson(u, v) for u, v in zip(nist_plot_dict['pred_list'], nist_plot_dict['exp_list'])]
        sl_nn = [utils.pearson(u, v) for u, v in zip(nist_plot_dict_nn['pred_list'], nist_plot_dict_nn['exp_list'])]
        sl_dft_er = [utils.pearson(u, v) for u, v in zip(nist_plot_dict_er['sim_list'], nist_plot_dict_er['exp_list'])]
        sl_gpr_er = [utils.pearson(u, v) for u, v in zip(nist_plot_dict_er['pred_list'], nist_plot_dict_er['exp_list'])]
        
        plot_hist([sl_dft, sl_nn, sl_gpr])
        
        plot_violin([sl_dft, sl_gpr, sl_dft_er, sl_gpr_er], 
           xlabel='Methods',
           ylabel='Similarity Measure', 
           labels=['DFT', 'GPR', 'DFT/curated', 'GPR/curated'])
        
        iden_matrix_quasi = GPUCalculator().pearson_matrix_gpu(nist_plot_dict['pred_list'], nist_plot_dict['exp_list'])
        iden_matrix_dft = GPUCalculator().pearson_matrix_gpu(nist_plot_dict['sim_list'], nist_plot_dict['exp_list'])
        iden_matrix_nn = GPUCalculator().pearson_matrix_gpu(nist_plot_dict_nn['pred_list'], nist_plot_dict_nn['exp_list'])
        
        plot_roc_list([iden_matrix_quasi, iden_matrix_nn, iden_matrix_dft], ['GPR', 'NN', 'DFT'])
        plot_roc_list([iden_matrix_quasi, iden_matrix_dft], ['GPR', 'DFT'])
    
        iden_matrix_quasi_er = GPUCalculator().pearson_matrix_gpu(nist_plot_dict_er['pred_list'], nist_plot_dict_er['exp_list'])
        iden_matrix_dft_er = GPUCalculator().pearson_matrix_gpu(nist_plot_dict_er['sim_list'], nist_plot_dict_er['exp_list'])
        plot_roc_list([iden_matrix_quasi, iden_matrix_dft, iden_matrix_quasi_er, iden_matrix_dft_er], ['GPR', 'DFT', 'GPR/curated', 'DFT/curated'])
    
    dft_data_plot()

def set_natcomsci_style():
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'Helvetica'], # 使用要求字体
        'font.size': 6,                           # 正文文字大小 (pt)
        'axes.titlesize': 6,                      # 子图标题大小
        'axes.labelsize': 6,                      # 坐标轴标签大小
        'xtick.labelsize': 6,                     # X轴刻度标签，可略小于正文
        'ytick.labelsize': 6,                     # Y轴刻度标签
        'legend.fontsize': 6,                     # 图例字体大小
        'figure.dpi': 300,                        # 导出时DPI，影响嵌入的位图
        'savefig.dpi': 300,
        'pdf.fonttype': 42,                       # 确保字体嵌入PDF并可编辑 (重要!)
        'ps.fonttype': 42,
        'figure.constrained_layout.use': True     # 自动调整布局，避免标签重叠
    })
    
def plot_heatmap():
    set_natcomsci_style()
    
    def diff_value_gen(sim_data, exp_data, pred_data):
        dfsel = []
        dfpel = []
        mae_list = []
        mae_list_pred = []
        for sim_ir, exp_ir, pred_ir in zip(sim_data, exp_data, pred_data):
            diff_se = exp_ir - sim_ir
            diff_pe = exp_ir - pred_ir
            dfsel.append(diff_se)
            dfpel.append(diff_pe)
            mae_list.append([abs(s - e) for s, e in zip(sim_ir, exp_ir)])
            mae_list_pred.append([abs(p - e) for p, e in zip(pred_ir, exp_ir)])
        return dfsel, dfpel, mae_list, mae_list_pred
    
    def ir_spectra_heatmaps(ir_list, x=None, width_cm=8, height_cm=5, save_name="ir_heatmap"):
        if x is None:
            x = utils.load_npy('wavenumber_550-3846-4') 
        
        ir_array = np.array(ir_list)
        
        # 厘米转英寸
        width_inch = width_cm / 2.54
        height_inch = height_cm / 2.54
        
        fig, ax = plt.subplots(figsize=(width_inch, height_inch), dpi=600)
        
        cax = ax.imshow(ir_array, aspect='auto', cmap='rainbow', 
                        extent=[x[0], x[-1], len(ir_list), 0], 
                        vmin=-0.5, vmax=0.5)
        
        # 设置坐标轴标签（Arial字体，7pt，不加粗）
        ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=6)
        ax.set_ylabel('Sample Index', fontsize=6)
        
        # 设置刻度标签
        ax.tick_params(axis='both', labelsize=6)
        
        # 颜色条设置
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Intensity (a.u.)', fontsize=6)
        cbar.ax.tick_params(labelsize=6)
        
        # 调整布局
        # plt.tight_layout()
        
        # 保存PDF
        save_path = f"{save_name}.pdf"
        fig.savefig(save_path, format='pdf', dpi=600)#bbox_inches='tight', pad_inches=0.05,
        print(f"已保存: {save_path}")
        
        plt.show()
        plt.close(fig)
        
    def mae_plot(mael_list, label_list, x=None):
        if x is None:
            x = utils.load_npy('wavenumber_550-3846-4') 
        
        plt.figure(figsize=(10, 6), dpi=600)  
        
        
        for i, (mae_list, label) in enumerate(zip(mael_list, label_list)):
            color = plt.cm.tab10(i)
            mae_array = np.array(mae_list)
            ave_mae_array = np.mean(mae_array, axis=0)
            plt.plot(x, ave_mae_array, label=label, color=color)
            
        plt.xlabel('Wavenumber (cm$^{-1}$)')
        plt.ylabel('Mean Absolute Error')
        plt.ylim(0, 0.4)
        plt.legend()
        plt.show()
        
    def heatmap_plot(data, label_list, name_list=['heatmap1, heatmap2']):
        dfsel, dfpel, mae_list, mae_list_pred = diff_value_gen(data['sim_list'], data['exp_list'], data['pred_list'])
        ir_spectra_heatmaps(dfsel, x=None, width_cm=8, height_cm=5, save_name=name_list[0])#"ir_heatmap1")
        ir_spectra_heatmaps(dfpel, x=None, width_cm=8, height_cm=5, save_name=name_list[1])#"ir_heatmap2") 
        mae_plot([mae_list, mae_list_pred], label_list)
        return (dfsel, dfpel, mae_list, mae_list_pred)
        
    nist_plot_dict = utils.load_dict('unique_nist_plot_dict')
    nist_plot_dict_er = utils.load_dict('unique_nist_plot_dict_er')
    
    heatmap_plot(nist_plot_dict, ['DFT', 'AUG'], ['dft_heatmap', 'aug_heatmap'])
    heatmap_plot(nist_plot_dict_er, ['DFT', 'AUG'],['dfter_heatmap', 'auger_heatmap'])
 
def plot_colored_PIDD():
    set_natcomsci_style()
    
    def get_row_max(matrix):
        """
        Calculates the maximum value in each row of a matrix, excluding the number 1.

        Args:
            matrix: A list of lists representing the matrix.

        Returns:
            A list containing the maximum value for each row, excluding 1.
            If a row contains only 1s or is empty, the corresponding element in the
            result will be None.
        """
        max_values = []
        for row in matrix:
            filtered_row = [x for x in row if x != 1]
            if filtered_row:
                max_values.append(max(filtered_row))
            else:
                max_values.append(None)  # Or any other suitable indicator for no valid max
        return max_values

    def figure_error_mol(data_train, data_test, data_test_pred, data_all, data_all_er,
                     width_cm=8, height_cm=5, save_prefix="error_mol_scatter"):
        
        vali_dict = utils.load_dict('unique_ggnd_vali_dict')
        vali_dict_train = utils.load_dict('unique_ggnd_vali_dict_train')
        
        vali_max_sim_row_list = []
        vali_dvs_quasi_list = []
        vali_dvs_dft_list = []
        vali_id_list = []
        for key in vali_dict.keys():
            id_list_vali = vali_dict[key]['id_list']
            similarity_matrix_vali = GPUCalculator().pearson_matrix_gpu(vali_dict[key]['sim_list'],
                                                                        vali_dict_train[key]['sim_list'])
            max_similarity_row_max_vali = get_row_max(similarity_matrix_vali)
            iden_matrix_quasi_vali = GPUCalculator().pearson_matrix_gpu(vali_dict[key]['exp_list'], vali_dict[key]['pred_list'])
            iden_matrix_dft_vali = GPUCalculator().pearson_matrix_gpu(vali_dict[key]['exp_list'], vali_dict[key]['sim_list'])
            dvs_quasi_vali = np.diagonal(iden_matrix_quasi_vali)
            dvs_dft_vali = np.diagonal(iden_matrix_dft_vali)
            vali_max_sim_row_list.extend(max_similarity_row_max_vali)
            vali_dvs_quasi_list.extend(dvs_quasi_vali)
            vali_dvs_dft_list.extend(dvs_dft_vali)
            vali_id_list.extend(id_list_vali)
            
        # 四象限着色
        similarity_matrix = GPUCalculator().pearson_matrix_gpu(data_test['sim_list'], data_train['sim_list'])
        max_similarity_row_max = get_row_max(similarity_matrix)
        
        iden_matrix_quasi = GPUCalculator().pearson_matrix_gpu(data_test['exp_list'], data_test_pred)
        iden_matrix_dft = GPUCalculator().pearson_matrix_gpu(data_test['exp_list'], data_test['sim_list'])
        dvs_quasi = np.diagonal(iden_matrix_quasi)
        dvs_dft = np.diagonal(iden_matrix_dft)
        
        all_max_sim = np.concatenate((vali_max_sim_row_list, max_similarity_row_max))
        all_dvs_quasi = np.concatenate((vali_dvs_quasi_list, dvs_quasi))
        all_dvs_dft = np.concatenate((vali_dvs_dft_list, dvs_dft))
        
        def find_iminol_ids(id_list, id_list_er, id_list_error):
            set2 = set(id_list_er)
            set3 = set(id_list_error)
            iminol_ids = [id for id in id_list if id not in set2 and id not in set3]
            return iminol_ids
        
        iminol_ids = find_iminol_ids(data_all['id_list'], data_all_er['id_list'], ERROR_IDS)
        
        colors_dis = []
        marker_dis = []
        
        all_id_list = np.concatenate((vali_id_list, data_test['id_list']))
        
        for idn in all_id_list:#data_test['id_list']:
            if idn in iminol_ids:
                colors_dis.append('green')
                marker_dis.append('s')
            elif idn in ERROR_IDS:
                colors_dis.append('red')
                marker_dis.append('o')
            else:
                colors_dis.append('blue')
                marker_dis.append('^')

        # 绘图函数（通用）
        def plot_scatter_figure(x_data, y_data, y_label, save_suffix):
            width_inch = width_cm / 2.54
            height_inch = height_cm / 2.54
            
            fig, ax = plt.subplots(figsize=(width_inch, height_inch), dpi=600)
            
            import pandas as pd
            color_to_marker = {'red': 'X', 'green': 's', 'blue': '^'}
            
            df = pd.DataFrame({
                'x': x_data,
                'y': y_data,
                'color': colors_dis
            }).dropna()
            
            for color, group in df.groupby('color'):
                ax.scatter(group['x'], group['y'],
                           color=color,
                           marker=color_to_marker[color],
                           alpha=0.6,
                           s=10,
                           edgecolors='black',
                           linewidths=0.3)
            
            ax.set_ylabel(y_label, fontsize=6)
            ax.set_xlabel('Max Similarity Score', fontsize=6)
            
            # 虚线设置
            ax.plot([0, 1], [0.8, 0.8], color='black', alpha=0.8, linestyle='--', linewidth=0.8)
            ax.plot([0.8, 0.8], [0, 1], color='black', alpha=0.8, linestyle='--', linewidth=0.8)
            ax.set_xlim(0.0, 1.01)
            ax.set_ylim(0.0, 1.01)
            
            # 网格设置
            ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
            
            # 创建自定义图例
            legend_handles = [
                plt.Line2D([0], [0], marker='X', color='w', label='Error',
                           markerfacecolor='red', markersize=6, markeredgecolor='black', markeredgewidth=0.3),
                plt.Line2D([0], [0], marker='s', color='w', label='Iminol',
                           markerfacecolor='green', markersize=6, markeredgecolor='black', markeredgewidth=0.3),
                plt.Line2D([0], [0], marker='^', color='w', label='Normal',
                           markerfacecolor='blue', markersize=6, markeredgecolor='black', markeredgewidth=0.3)
            ]
            
            # 添加图例（无边框）
            ax.legend(handles=legend_handles, title="Molecular Status", 
                      fontsize=6, title_fontsize=6, frameon=False)
            
            # 设置刻度标签
            ax.tick_params(axis='both', labelsize=6)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存PDF
            save_path = f"{save_prefix}_{save_suffix}.pdf"
            fig.savefig(save_path, format='pdf', dpi=600)#bbox_inches='tight',pad_inches=0.05, 
            print(f"已保存: {save_path}")
            
            plt.show()
            plt.close(fig)
          
        plot_scatter_figure(all_max_sim, all_dvs_quasi, 
                           'Matching Rate (Quasi-exp)', 'quasi')

        plot_scatter_figure(all_max_sim, all_dvs_dft, 
                           'Matching Rate (DFT)', 'dft')
        
        # 统计部分（保留原有统计）
        def count_quadrants(x_values, y_values):
            quadrant_counts = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
            for x, y in zip(x_values, y_values):
                if x >= 0.8 and y >= 0.8:
                    quadrant_counts['Q1'] += 1
                elif x < 0.8 and y >= 0.8:
                    quadrant_counts['Q2'] += 1
                elif x < 0.8 and y < 0.8:
                    quadrant_counts['Q3'] += 1
                elif x >= 0.8 and y < 0.8:
                    quadrant_counts['Q4'] += 1
            
            print("=" * 50)
            print("象限统计结果 (x=0.8, y=0.8为分界)")
            print("=" * 50)
            print(f"总点数: {len(x_values)}")
            print(f"Q1 (右上): {quadrant_counts['Q1']} 个点")
            print(f"Q2 (左上): {quadrant_counts['Q2']} 个点") 
            print(f"Q3 (左下): {quadrant_counts['Q3']} 个点")
            print(f"Q4 (右下): {quadrant_counts['Q4']} 个点")
            print("-" * 50)
            
            return quadrant_counts
        
        def count_by_status(x_values, y_values, colors_dis):
            status_quadrants = {
                'red': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0},
                'green': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0},
                'blue': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
            }
            
            for x, y, color in zip(x_values, y_values, colors_dis):
                if x >= 0.8 and y >= 0.8:
                    quadrant = 'Q1'
                elif x < 0.8 and y >= 0.8:
                    quadrant = 'Q2'
                elif x < 0.8 and y < 0.8:
                    quadrant = 'Q3'
                elif x >= 0.8 and y < 0.8:
                    quadrant = 'Q4'
                else:
                    continue
                status_quadrants[color][quadrant] += 1
                
            print("\n按分子状态统计:")
            print("-" * 30)
            for color, counts in status_quadrants.items():
                status_name = {'red': 'Error', 'green': 'Iminol', 'blue': 'Normal'}[color]
                print(f"{status_name}分子:")
                print(f"  Q1: {counts['Q1']}, Q2: {counts['Q2']}, Q3: {counts['Q3']}, Q4: {counts['Q4']}")
                print(f"  总计: {sum(counts.values())}")
                print()    
            
            return status_quadrants
        
        total_counts = count_quadrants(all_max_sim, all_dvs_dft)  
        status_counts = count_by_status(all_max_sim, all_dvs_dft, colors_dis)
        total_counts_e = count_quadrants(all_max_sim, all_dvs_quasi)  
        status_counts_e = count_by_status(all_max_sim, all_dvs_quasi, colors_dis)
        
    nist_plot_dict = utils.load_dict('unique_nist_plot_dict')
    nist_plot_dict_er = utils.load_dict('unique_nist_plot_dict_er')
    
    vali_plot_dict = utils.load_dict('unique_vali_plot_dict')
    vali_plot_dict_er = utils.load_dict('unique_vali_plot_dict_er')
    
    train_data = utils.load_dict('unique_train_data_no_Dsub')
    test_data = utils.load_dict('unique_test_data_no_Dsub')
    cleaned_train_data = utils.load_dict('unique_clean_train_data_no_error_and_iminol')
    cleaned_test_data = utils.load_dict('unique_clean_test_data_no_error_and_iminol')
    
    vali_dict_train = utils.load_dict('unique_ggnd_vali_dict_train')
    vali_dict_train_er = utils.load_dict('unique_ggnder_vali_dict_train')

    test_pred_ggnd = utils.load_dict('unique_ggnd_test_pred')
    test_pred_ggnder = utils.load_dict('unique_ggnder_test_pred')

    train_data_dict ={
        'id_list': train_data.ids,
        'sim_list': train_data.features,
        'exp_list': train_data.targets,
        'mw_int_list': train_data.mws,
        'smiles_list': train_data.smiles
    }
    
    test_data_dict ={
        'id_list': test_data.ids,
        'sim_list': test_data.features,
        'exp_list': test_data.targets,
        'mw_int_list': test_data.mws,
        'smiles_list': test_data.smiles
    }
    
    cleaned_test_data_dict ={
        'id_list': cleaned_test_data.ids,
        'sim_list': cleaned_test_data.features,
        'exp_list': cleaned_test_data.targets,
        'mw_int_list': cleaned_test_data.mws,
        'smiles_list': cleaned_test_data.smiles
    }
    
    figure_error_mol(train_data_dict, test_data_dict, test_pred_ggnd, nist_plot_dict, nist_plot_dict_er)   
    
def plot_fgs_iden():
    
    def max_value_indices_and_sorting_order(matrix):
        matrix = np.array(matrix)
        
        max_indices = []
        max_values = []
        diagonal_ranks = []
        
        for i in range(len(matrix)):
            row = matrix[i]
            
            max_value = np.max(row)
            max_values.append(max_value)
            
            diagonal_value = row[i]
            rank = np.argsort(-row)  
            rank = np.argsort(rank) + 1  
            diagonal_ranks.append(rank[i])
            
            max_idx = np.argmax(row)
            max_indices.append(max_idx)
            
        return max_values, diagonal_ranks, max_indices
    
    def fgs_ana(ids, max_ids, all_id):
        
        def id_fgs_map_all_gen():
            nist_data = data_loader.load_nist_data()
            db_data = data_loader.load_130k_data()
            
            id_list = nist_data['id_list']
            fgs_list = nist_data['fgs_dict']
            db_id_list = db_data['id_list']
            db_fgs_list = db_data['fgs_dict']
            
            idn_fgs_map_all = {}
            for idn, fgs in zip(id_list, fgs_list):
                if idn not in idn_fgs_map_all:
                    idn_fgs_map_all[idn] = fgs
                    
            for idn, fgs in zip(db_id_list, db_fgs_list):
                if idn not in idn_fgs_map_all:
                    idn_fgs_map_all[idn] = fgs
                    
            print(len(idn_fgs_map_all))
            
            utils.save_dict(idn_fgs_map_all, 'idn_fgs_map_all')
        
        id_fgs_map_all_gen()
        all_id_fgs_map = utils.load_dict('idn_fgs_map_all')
        same_result = {}
        sub_result = {}
        super_result = {}
        for idn, max_id in zip(ids, max_ids):
            id_com = all_id[max_id]
            fg_list = all_id_fgs_map[idn]
            fg_list_com = all_id_fgs_map[id_com]
            for fg in fg_list:
                if fg in fg_list_com:
                    if fg in same_result:
                        same_result[fg] += 1
                    else:
                        same_result[fg] = 1
                else:
                    if fg in sub_result:
                        sub_result[fg] += 1
                    else:
                        sub_result[fg] = 1
            for fg in fg_list_com:
                if fg not in fg_list:
                    if fg in super_result:
                        super_result[fg] += 1
                    else:
                        super_result[fg] = 1
                        
        fgs_com_list = []
        f1_score_list = []
        recall_list = []
        precision_list = []
        for key in same_result:
            if key not in super_result:
                super_result[key] = 0
            if key not in sub_result:
                sub_result[key] = 0
            precision = same_result[key] / (same_result[key] + super_result[key])    
            recall = same_result[key] / (same_result[key] + sub_result[key])    
            f1_score = (2 * same_result[key]) / ( 2 * same_result[key] + sub_result[key] + super_result[key])
            fgs_com_list.append(key)
            f1_score_list.append(f1_score)
            recall_list.append(recall)
            precision_list.append(precision)
            
        fgs_analysis_result = [fgs_com_list, f1_score_list, precision_list, recall_list]
     
        return fgs_analysis_result
    
    def extract_and_sort_f1_scores_with_element_limited(perf_dict):
        """
        Extract F1 scores from multiple performance dictionaries and sort by custom F1 scores.
        
        Args:
            agg_perf_dict: Dictionary with keys 'fg' and nested 'm-label' containing 'F1'
            mlp_perf_dict: Dictionary with keys 'fg' and 'vali f1'
            fcg_perf_dict: Dictionary with keys 'fg' and 'F1'
            perf_dict: Dictionary with keys 'fg' and 'F1' (custom scores to sort by)
        
        Returns:
            Tuple of (fg_list, agg_f1_list, mlp_f1_list, fcg_f1_list, my_f1_list)
            All lists are sorted by perf_dict's F1 scores in descending order
        """
        jj_perf_dict = external_data.JJ_PERFORMANCE
        
        common_fgs = set(perf_dict.keys())
        for d in [jj_perf_dict]:
            common_fgs.intersection_update(d.keys())
        
        # Initialize storage for complete data only
        complete_data = []
        
        # Collect data only for functional groups present in all dictionaries
        for fg in common_fgs:
            # Get all values, skip if any are None
            jj_f1 = jj_perf_dict[fg].get('F1')
            my_f1 = perf_dict[fg].get('F1')
            
            if None not in [jj_f1, my_f1]:
                complete_data.append((fg, jj_f1, my_f1))
        
        # Sort by custom F1 scores (perf_dict) in descending order
        complete_data.sort(key=lambda x: x[2], reverse=True)
        
        # Unzip the sorted data
        if complete_data:
            fg_list, jj_f1_list, my_f1_list = zip(*complete_data)
            print('our_ave_F1: ', np.mean(my_f1_list))
            print('jj_ave_F1: ', np.mean(jj_f1_list))
            
            return (
                list(fg_list),
                list(my_f1_list),
                list(jj_f1_list)
            )
        else:
            return ([], [], [])

    def extract_and_sort_f1_scores(perf_dict):
        """
        Extract F1 scores from multiple performance dictionaries and sort by custom F1 scores.
        
        Args:
            agg_perf_dict: Dictionary with keys 'fg' and nested 'm-label' containing 'F1'
            mlp_perf_dict: Dictionary with keys 'fg' and 'vali f1'
            fcg_perf_dict: Dictionary with keys 'fg' and 'F1'
            perf_dict: Dictionary with keys 'fg' and 'F1' (custom scores to sort by)
        
        Returns:
            Tuple of (fg_list, agg_f1_list, mlp_f1_list, fcg_f1_list, my_f1_list)
            All lists are sorted by perf_dict's F1 scores in descending order
        """
        agg_perf_dict = external_data.AGG_PERFORMANCE
        mlp_perf_dict = external_data.MLP_PERFORMANCE
        fcg_perf_dict = external_data.FCG_PERFORMANCE
        
        common_fgs = set(perf_dict.keys())
        for d in [agg_perf_dict, mlp_perf_dict, fcg_perf_dict]:
            common_fgs.intersection_update(d.keys())
        
        # Initialize storage for complete data only
        complete_data = []
        
        # Collect data only for functional groups present in all dictionaries
        for fg in common_fgs:
            # Get all values, skip if any are None
            agg_f1 = agg_perf_dict[fg].get('m-Label', {}).get('F1')
            mlp_f1 = mlp_perf_dict[fg].get('Validation set F1')
            fcg_f1 = fcg_perf_dict[fg].get('F1')
            my_f1 = perf_dict[fg].get('F1')
            
            if None not in [agg_f1, mlp_f1, fcg_f1, my_f1]:
                complete_data.append((fg, agg_f1, mlp_f1, fcg_f1, my_f1))
        
        # Sort by custom F1 scores (perf_dict) in descending order
        complete_data.sort(key=lambda x: x[4], reverse=True)
        
        # Unzip the sorted data
        if complete_data:
            fg_list, agg_f1_list, mlp_f1_list, fcg_f1_list, my_f1_list = zip(*complete_data)
            print('our_ave_F1: ', np.mean(my_f1_list))
            print('agg_ave_F1: ', np.mean(agg_f1_list))
            print('mlp_ave_F1: ', np.mean(mlp_f1_list))
            print('fcg_ave_F1: ', np.mean(fcg_f1_list))
            
            return (
                list(fg_list),
                list(my_f1_list),
                list(agg_f1_list),
                list(mlp_f1_list),
                list(fcg_f1_list)
            )
        else:
            return ([], [], [], [], [])
  
    def plot_heatmap_max_T(sort_result, model='All'):
        import seaborn as sns
        if model == 'All':
            fg_list, my_f1_list, agg_f1_list, mlp_f1_list, fcg_f1_list = sort_result
            data = np.array([my_f1_list, agg_f1_list, mlp_f1_list, fcg_f1_list]).T  # 转置后变为 N行 x 4列
            row_labels = fg_list  # 现在行标签是功能基团
            col_labels = ['Quasi-exp IR', 'AggMapNet', 'MLP', 'Fcg-Former']  # 现在列标签是方法
        
        elif model == 'EL':
            fg_list, my_f1_list, jj_f1_list = sort_result
            data = np.array([my_f1_list, jj_f1_list]).T 
            row_labels = fg_list
            col_labels = ['Quasi-exp IR', 'PatchBasedSelfAttention']
            
        # 创建标注矩阵（全部数值+旋转90°）
        annot_matrix = np.array([[f"{val:.2f}" for val in row] for row in data])
        
        # 标记每行最大值位置（原每列最大值）
        max_mask = np.zeros_like(data, dtype=bool)
        for i in range(data.shape[0]):
            max_val = np.max(data[i, :])
            max_mask[i, :] = (data[i, :] == max_val)
        
        plt.figure(figsize=(10, 6), dpi=600)  # 调整图形尺寸适应新布局
        
        # 第一步：绘制基础热力图
        heatmap = sns.heatmap(
            data,
            annot=False,
            fmt="",
            cmap="coolwarm",
            xticklabels=col_labels,
            yticklabels=row_labels,
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={
                "label": "F1 Score",
                "shrink": 0.8,
            }
        )

        # 获取 colorbar 并设置标签加粗
        cbar = heatmap.collections[0].colorbar
        cbar.ax.yaxis.label.set_weight("bold") 
        
        # 第二步：手动添加旋转标注
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = "gold" if max_mask[i, j] else "black"
                weight = "bold" if max_mask[i, j] else "normal"
                heatmap.text(
                    j + 0.5,
                    i + 0.5,
                    annot_matrix[i, j],
                    ha="center",
                    va="center",
                    # rotation=90,
                    color=color,
                    weight=weight,
                    fontsize=14
                )
        
        # 第三步：突出最大值单元格
        for i, j in zip(*np.where(max_mask)):
            heatmap.add_patch(plt.Rectangle(
                (j, i), 1, 1,
                fill=False,
                edgecolor="gold",
                lw=2
            ))
        
        # 调整标签样式
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.xlabel("Methods", fontweight="bold")
        plt.ylabel("Functional Groups", fontweight="bold")
        
        plt.tight_layout()
        plt.show() 
    
    def plot_radar_max_T(sort_result, model='All', save_name="radar_chart"):
        set_natcomsci_style()
        from matplotlib.patches import Patch
        """
        绘制雷达图比较不同方法在各功能基团上的表现
        
        参数:
            sort_result: 排序后的结果 (格式与热力图函数一致)
            model: 'All' 或 'EL'，指定显示哪些模型
        """
        # 数据准备 (与热力图函数保持一致)
        if model == 'All':
            fg_list, my_f1_list, agg_f1_list, mlp_f1_list, fcg_f1_list = sort_result
            data = np.array([my_f1_list, agg_f1_list, mlp_f1_list, fcg_f1_list]).T
            method_labels = ['Quasi-exp IR', 'AggMapNet', 'MLP', 'Fcg-Former']
            colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']  # 不同方法的颜色
        elif model == 'EL':
            fg_list, my_f1_list, jj_f1_list = sort_result
            data = np.array([my_f1_list, jj_f1_list]).T
            method_labels = ['Quasi-exp IR', 'PatchBasedSelfAttention']
            colors = ['#d62728', '#1f77b4']
        
        width_cm=6
        height_cm=6 
        width_inch = width_cm / 2.54
        height_inch = height_cm / 2.54
        
        # 雷达图参数设置
        num_vars = len(fg_list)  # 变量数（功能基团数）
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 创建极坐标图
        fig, ax = plt.subplots(figsize=(width_inch, height_inch), 
                              subplot_kw=dict(polar=True),
                              dpi=600)
        
        # 绘制每个方法的数据
        for idx, method_data in enumerate(data.T):
            values = method_data.tolist()
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, color=colors[idx], linewidth=1.0, label=method_labels[idx])
            ax.fill(angles, values, color=colors[idx], alpha=0.1)
        
        # 标记每个功能基团的最高分方法（可选，如果太拥挤可以去掉）
        max_indices = np.argmax(data, axis=1)
        for i, (angle, fg) in enumerate(zip(angles[:-1], fg_list)):
            best_method_idx = max_indices[i]
            best_value = data[i, best_method_idx]
            ax.plot([angle], [best_value], 
                    marker='o', 
                    markersize=3, 
                    color=colors[best_method_idx],
                    markeredgecolor='white',
                    markeredgewidth=0.3)
        
            # 显示数值（减小字体，避免重叠）
            ax.text(angle, 1.05, f'{best_value:.2f}',
                    ha='center', va='center',
                    color=colors[best_method_idx],
                    fontsize=5)  # 移除bold
        
        # 设置极坐标轴
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 设置角度网格标签（使用Arial字体，减小字号）
        ax.set_thetagrids(np.degrees(angles[:-1]), labels=fg_list)
        ax.tick_params(axis='x', labelsize=6, pad=10)  # 减小padding
        
        # 设置径向轴
        ax.set_rlabel_position(0)
        plt.yticks(np.linspace(0, 1, 5), color="black", size=6)
        ax.tick_params(axis='y', labelsize=6)
        plt.ylim(0, 1)
        
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        # 设置刻度线宽
        ax.tick_params(width=0.5, length=2)
        
        # 添加图例（简化版，放在图形内部）
        legend = ax.legend(loc='upper right', 
                          bbox_to_anchor=(1.1, 1.0),  # 靠近图形
                          fontsize=6, 
                          frameon=True,
                          fancybox=False,
                          edgecolor='black',
                          facecolor='white',
                          framealpha=0.9,
                          borderpad=0.5)
        legend.get_frame().set_linewidth(0.5)
        
        # 移除图例文字加粗
        for text in legend.get_texts():
            text.set_fontsize(6)
            text.set_fontname('Arial')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存PDF
        save_path = f"{save_name}.pdf"
        fig.savefig(save_path, format='pdf', bbox_inches='tight', 
                    pad_inches=0.05, dpi=600)
        print(f"已保存: {save_path}")
        
        plt.show()
        plt.close(fig)
        
    def fg_compare_heatmap(ana_fgs_r1, model='All', save_name='radar_chart'):
        
        fgs_com_list, f1_score_list, precision_list, recall_list = ana_fgs_r1
        
        perf_dict = {
            fgs: {
                "F1": f1,
                "Precision": precision,
                "Recall": recall
            }
            for fgs, f1, precision, recall in zip(fgs_com_list, f1_score_list, precision_list, recall_list)
        }
        
        if model == 'All':
            sort_result = extract_and_sort_f1_scores(perf_dict)
            
        elif model == 'EL':
            sort_result = extract_and_sort_f1_scores_with_element_limited(perf_dict)
            
        plot_radar_max_T(sort_result, model=model, save_name=save_name)
    
    def max_value_indices_and_sorting_order_single_row(row_matrix, original_index):
        """
        Analyzes a single row (or a matrix containing a single row) of correlation coefficients.

        Args:
            row_matrix (np.ndarray): A NumPy array representing a single row of correlation
                                     coefficients. This could be of shape (N,) or (1, N).
            original_index (int): The index 'i' of the target from xtb_test_raw.targets
                                  that this row corresponds to. This is crucial for
                                  calculating the diagonal rank correctly.

        Returns:
            tuple: A tuple containing:
                - max_value (float): The maximum correlation value in the row.
                - diagonal_rank (int): The rank of the correlation with itself (at original_index)
                                       within this row.
                - max_idx (int): The index of the maximum correlation value in the row.
        """
        # Ensure row_matrix is a 1D array for easier processing
        if row_matrix.ndim > 1:
            row = row_matrix.flatten()
        else:
            row = row_matrix

        max_value = np.max(row)
        max_idx = np.argmax(row)

        # Calculate diagonal rank. We need to know which element corresponds to the "diagonal".
        # In your loop, `exp_ir` is the i-th element, and `xtb_pred_nistpdb` is the whole set.
        # So, the "diagonal" element for `exp_ir` is the correlation of `exp_ir` with the
        # i-th element of `xtb_pred_nistpdb`.
        # Assuming the `iden_matrix` result from `pearson_matrix_gpu` for `[exp_ir,]`
        # will have its columns aligned with `xtb_pred_nistpdb`.
        
        # Check if original_index is within the bounds of the current row.
        # This assumes that xtb_pred_nistpdb has at least `original_index + 1` elements.
        if original_index < len(row):
            # diagonal_value = row[original_index]
            
            # Calculate rank: argsort gives indices of sorted values.
            # np.argsort(-row) gives indices that would sort in descending order.
            # Then, applying argsort again gives the rank of each element.
            ranks = np.argsort(np.argsort(-row)) + 1
            diagonal_rank = ranks[original_index]
        else:
            # If the original_index is out of bounds for the current row,
            # it means there's no corresponding "diagonal" element in this specific correlation row.
            # This might happen if len(xtb_pred_nistpdb) is smaller than original_index,
            # or if the `pearson_matrix_gpu` result is not as expected.
            diagonal_rank = None # Or handle as an error, depending on your logic

        return max_value, diagonal_rank, max_idx
    
    def iden_matrix_process_single(data, pred_db, pearson_matrix_name):
        all_max_values = []
        all_diagonal_ranks = []
        all_max_indices = []
        
        pearson_matrix = np.zeros((len(data), len(pred_db)))
    
        for i, exp_ir in enumerate(data):
            iden_matrix_row = GPUCalculator.pearson_matrix_gpu([exp_ir,], pred_db)[0]
            pearson_matrix[i, :] = iden_matrix_row
            
            max_value, diagonal_rank, max_idx = \
                max_value_indices_and_sorting_order_single_row(iden_matrix_row, i)
            
            all_max_values.append(max_value)
            all_diagonal_ranks.append(diagonal_rank)
            all_max_indices.append(max_idx)
        
        utils.save_dict(pearson_matrix, pearson_matrix_name)
        
        return all_max_values, all_diagonal_ranks, all_max_indices
    
    def classify_molecules_rdkit(id_list, smiles_list, sim_list):
        from rdkit import Chem
        classification = {
            "CH": {"ids": [], "sim": []},
            "CHO": {"ids": [], "sim": []},
            "CHON": {"ids": [], "sim": []}
        }
        
        for id_, smiles, spec in zip(id_list, smiles_list, sim_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(id_, "mol not found")
                    continue
                    
                elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
                
                if 'N' in elements:
                    classification["CHON"]["ids"].append(id_)
                    classification["CHON"]["sim"].append(spec)
                elif 'O' in elements:
                    classification["CHO"]["ids"].append(id_)
                    classification["CHO"]["sim"].append(spec)
                else:
                    classification["CH"]["ids"].append(id_)
                    classification["CH"]["sim"].append(spec)
            except:
                print(id_, "not classify")
                continue
        
        return classification    
    
    def id_fgs_map_all_gen():
        nist_data = data_loader.load_nist_data()
        db_data = data_loader.load_130k_data()
        
        id_list = nist_data['id_list']
        fgs_list = nist_data['fgs_dict']
        db_id_list = db_data['id_list']
        db_fgs_list = db_data['fgs_dict']
        
        idn_fgs_map_all = {}
        for idn, fgs in zip(id_list, fgs_list):
            if idn not in idn_fgs_map_all:
                idn_fgs_map_all[idn] = fgs
                
        for idn, fgs in zip(db_id_list, db_fgs_list):
            if idn not in idn_fgs_map_all:
                idn_fgs_map_all[idn] = fgs
        
        utils.save_dict(idn_fgs_map_all, 'idn_fgs_map_all')
    
    def fgs_ana_elements_limited(ids, max_ids, all_id):
        same_result = {}
        sub_result = {}
        super_result = {}
        idn_fgs_map_all = utils.load_dict('idn_fgs_map_all')
        for idn, max_id in zip(ids, max_ids):
            id_com = all_id[max_id]
            fg_list = idn_fgs_map_all[idn]
            fg_list_com = idn_fgs_map_all[id_com]
            for fg in fg_list:
                if fg in fg_list_com:
                    if fg in same_result:
                        same_result[fg] += 1
                    else:
                        same_result[fg] = 1
                else:
                    if fg in sub_result:
                        sub_result[fg] += 1
                    else:
                        sub_result[fg] = 1
            for fg in fg_list_com:
                if fg not in fg_list:
                    if fg in super_result:
                        super_result[fg] += 1
                    else:
                        super_result[fg] = 1
        fgs_com_list = []
        for key in same_result:
            if key not in super_result:
                super_result[key] = 0
            if key not in sub_result:
                sub_result[key] = 0
            fgs_com_list.append(key)
            
        fgs_analysis_result = [fgs_com_list, same_result, super_result, sub_result]
        
        return fgs_analysis_result
    
    def element_limited_fgs_dict_process(fgs_ana_result_dict):
        
        # fgs15 = ["Alkane", "Alkene", "Alkyne", "Aromatic", "Alcohol", "Ether", "Aldehyde", "Ketone", "Carboxylic Acid", "Ester", 
        #          "Amide", "Amine", "Nitrile", "Nitro", "Imine"]
        
        unique_fgs = set()
        # 遍历字典的所有键和子列表
        for key in fgs_ana_result_dict:
            fgs_info = fgs_ana_result_dict[key][0]
            if len(fgs_info) > 0:
                for fg in fgs_info:
                    unique_fgs.add(fg)  
                       
        same_dict ={fg: 0 for fg in list(unique_fgs)}
        super_dict = {fg: 0 for fg in list(unique_fgs)}
        sub_dict = {fg: 0 for fg in list(unique_fgs)}
        
        for key in fgs_ana_result_dict:
            same_result = fgs_ana_result_dict[key][1]
            super_result = fgs_ana_result_dict[key][2]
            sub_result = fgs_ana_result_dict[key][3]
            fgs_info = fgs_ana_result_dict[key][0]
            for fg in unique_fgs:
                if fg in fgs_info:
                    if fg not in super_result:
                        super_result[fg] = 0
                    if fg not in sub_result:
                        sub_result[fg] = 0
                    same_dict[fg] += same_result[fg]
                    super_dict[fg] += super_result[fg]
                    sub_dict[fg] += sub_result[fg]
                    
        fgs_com_list = []
        f1_score_list = []
        recall_list = []
        precision_list = []            
        for key in same_dict:
            precision = same_dict[key] / (same_dict[key] + super_dict[key])    
            recall = same_dict[key] / (same_dict[key] + sub_dict[key])    
            f1_score = (2 * same_dict[key]) / ( 2 * same_dict[key] + sub_dict[key] + super_dict[key])
            fgs_com_list.append(key)
            f1_score_list.append(f1_score)
            recall_list.append(recall)
            precision_list.append(precision)
            
        fgs_analysis_result = [fgs_com_list, f1_score_list, precision_list, recall_list]
        return fgs_analysis_result
    
    def plot_fgs_heatmap_el():
        # para: data_dict, db_id_pro, db_smiles_pro, db_test_pred
        # elements limited
        # train_size_er = 2138
        
        # all_id = np.concatenate((data_dict['id_list'][train_size_er:], 
        #                          data_dict['id_list'][:train_size_er],
        #                          db_id_pro))
        # all_smiles = np.concatenate((data_dict['smiles_list'][train_size_er:], 
        #                              data_dict['smiles_list'][:train_size_er],
        #                              db_smiles_pro))
        # all_pred = np.concatenate((data_dict['pred_list'][train_size_er:],
        #                            data_dict['pred_list'][:train_size_er],
        #                            db_test_pred))
        
        # class_test = classify_molecules_rdkit(data_dict['id_list'][train_size_er:], 
        #                                       data_dict['smiles_list'][train_size_er:], 
        #                                       data_dict['exp_list'][train_size_er:])
        
        # class_com = classify_molecules_rdkit(all_id, all_smiles, all_pred)
        
        # iden_matrix_dict = {}
        # mvs_dict = {}
        # drs_dict = {}
        # maxids_dict = {}
        # fgs_ana_result_dict_with_el = {}
        
        # for key in class_test: # ch, cho, chon
        #     iden_matrix_dict[key] = GPUCalculator().pearson_matrix_gpu(class_test[key]['sim'], class_com[key]['sim'])
        #     mvs, drs, maxids = max_value_indices_and_sorting_order(iden_matrix_dict[key])
        #     mvs_dict[key] = mvs
        #     drs_dict[key] = drs
        #     maxids_dict[key] = maxids
        #     fgs_ana_result_dict_with_el[key] = fgs_ana_elements_limited(class_test[key]['ids'], maxids, class_com[key]['ids'])

        # utils.save_dict(fgs_ana_result_dict_with_el, 'unique_fgs_count_result_dict_with_elements_limited_pro_ER')
        fgs_ana_result_dict_with_el = utils.load_dict('unique_fgs_count_result_dict_with_elements_limited_pro_ER')
        fgs_ana_result = element_limited_fgs_dict_process(fgs_ana_result_dict_with_el)
        fg_compare_heatmap(fgs_ana_result, model='EL')

    def detect_related_exp_ir_rank(exp_list, mw_list_int, mw_list_int_all, pearson_matrix_name):
        pearson_matrix = utils.load_dict(pearson_matrix_name)
        ranks = []
        for i, (exp_ir, mw) in enumerate(zip(exp_list, mw_list_int)):   
            same_mw_indices = [j for j, sim_mw in enumerate(mw_list_int_all) if sim_mw == mw]
            same_mw_values = pearson_matrix[i, same_mw_indices]
            diagonal_value = pearson_matrix[i, i]
            rank = np.sum(same_mw_values > diagonal_value) + 1
            if rank == len(same_mw_indices):
                rank = 21
            ranks.append(rank) 
        return ranks
    
    def plot_ranks(ranks_list, label, rank_top=20):
        suc_rate_list = []
        for rank in range(1, (rank_top + 1)):
            count = 0
            for value in ranks_list:
                if value <= rank:
                    count += 1
            suc_rate = count / len(ranks_list)
            suc_rate_list.append(suc_rate)
        
        print(suc_rate_list)
        
        x = np.arange(1, (rank_top + 1))
        
        plt.figure(figsize=(10, 6), dpi = 600)
        
        for i in range(1, 11):
            i_s = i/10
            plt.axhline(i_s, color='black', alpha=0.5, linestyle='--', linewidth=1) 
            
        plt.step(x, suc_rate_list, label=label, color='#33ABC1',linewidth=2, where='mid')
        plt.xlabel('Rank', fontweight='bold')
        plt.ylabel('Success Rate', fontweight='bold')
        plt.show()
 
    def suc_rate_rank_comparison(iden_rank_list, label_list, rank, i_min=2, i_max=11, 
                             width_cm=8, height_cm=5, save_name="suc_rate_rank_2"):
        # 设置全局样式
        set_natcomsci_style()
        
        def iden_success_rate_rank(iden_rank, rank=1):
            count = 0
            for value in iden_rank:
                if value <= rank:
                    count += 1
            suc_rate = count / len(iden_rank)
            return suc_rate
        
        def iden_list_rank(iden_rank, max_rank):
            suc_rate_list = []
            for r in range(1, max_rank):
                suc_rate = iden_success_rate_rank(iden_rank, r)
                suc_rate_list.append(suc_rate)
            return suc_rate_list
        
        x = np.arange(1, rank)
        
        # 创建图形（按Nature规范）
        width_inch = width_cm / 2.54
        height_inch = height_cm / 2.54
        fig, ax = plt.subplots(figsize=(width_inch, height_inch), dpi=600)
        
        # 添加水平参考线
        for i in range(i_min, i_max):
            i_s = i/10
            ax.axhline(i_s, color='black', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 绘制阶梯图
        colors_set = ['#84BA42', '#7ABBDB', '#682487','#A51C36', '#DBB428']
        color_index = 0
        for iden_rank, label in zip(iden_rank_list, label_list):
            suc_rate_list = iden_list_rank(iden_rank, rank)
            print(f"{label}: {suc_rate_list}")
            ax.step(x, suc_rate_list, label=label, 
                    color=colors_set[color_index % len(colors_set)],
                    linewidth=1.0, where='mid')
            color_index += 1
        
        # 设置坐标轴标签
        ax.set_xlabel('Rank', fontsize=6)
        ax.set_ylabel('Success Rate', fontsize=6)
        
        # 设置刻度
        ax.tick_params(axis='both', labelsize=5, width=0.5)
        
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        # 添加图例（有边框和底色，左上角）
        legend = ax.legend(loc='lower right',
                          fontsize=5,
                          frameon=True,
                          fancybox=False,
                          edgecolor='black',
                          facecolor='white',
                          framealpha=0.9,
                          borderpad=0.6,
                          handlelength=1.5,
                          handletextpad=0.5,
                          borderaxespad=0.5)
        legend.get_frame().set_linewidth(0.5)
        
        # 网格
        ax.grid(True, alpha=0.2, linewidth=0.3, linestyle='--')
        
        # 保存PDF
        save_path = f"{save_name}.pdf"
        fig.savefig(save_path, format='pdf', dpi=600)
        print(f"已保存: {save_path}")
        
        plt.show()
        plt.close(fig)

    def plot_natcomsci_performance(fgs_results_list, output_name='performance_comparison'):
        set_natcomsci_style()
        """
        绘制Nature Communications格式的四组对比柱状图
        
        Parameters:
        -----------
        fgs_analysis_result : list
            [fgs_com_list, f1_score_list, precision_list, recall_list]
            其中fgs_com_list是官能团列表，但此处我们固定使用['quasi', 'dft', 'quasi/curated', 'dft/curated']
            作为四个数据集的横坐标标签
        output_name : str
            输出文件名（不包含扩展名）
        """
        results = fgs_results_list
        
        avg_f1 = []
        avg_precision = []
        avg_recall = []
        
        for result in results:
            f1_list = result[1]      # F1 score列表
            precision_list = result[2]  # Precision列表
            recall_list = result[3]   # Recall列表
            
            avg_f1.append(np.mean(f1_list))
            avg_precision.append(np.mean(precision_list))
            avg_recall.append(np.mean(recall_list))

        # 固定数据集名称
        dataset_names = ['Quasi', 'DFT', 'Quasi/curated', 'DFT/curated']

        # 创建画布 - 8cm x 5cm (转换为英寸：1英寸=2.54cm)
        fig_width_cm = 8
        fig_height_cm = 5
        fig_width_inch = fig_width_cm / 2.54
        fig_height_inch = fig_height_cm / 2.54
        
        x = np.arange(len(dataset_names))
        width = 0.23
        colors = ['#4C72B0', '#55A868', '#C44E52']
        
        fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))
        
        bars1 = ax.bar(x - width, avg_f1, width, label='F1 Score', 
                   color=colors[0], edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, avg_precision, width, label='Precision', 
                       color=colors[1], edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x + width, avg_recall, width, label='Recall', 
                       color=colors[2], edgecolor='black', linewidth=0.5)
        
        # 设置标签
        ax.set_xlabel('Method', fontsize=6)
        ax.set_ylabel('Average Score', fontsize=6)
        
        # 设置x轴刻度
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, fontsize=5)
        ax.tick_params(axis='x', length=1)
        ax.tick_params(axis='y', length=1, labelsize=5)
        
        # 设置y轴范围（留出一些空间给数值标签）
        y_min = 0
        y_max = max(max(avg_f1), max(avg_precision), max(avg_recall)) * 1.15
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(0, 1.05, 0.2))
        ax.set_yticklabels([f'{i:.1f}' for i in np.arange(0, 1.05, 0.2)], fontsize=5)
        
        # 添加数值标签（显示平均值±标准差）
        def add_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', 
                           fontsize=5, fontweight='light')
        
        add_labels(bars1, avg_f1)
        add_labels(bars2, avg_precision)
        add_labels(bars3, avg_recall)
        
        # 添加图例
        ax.legend(loc='upper left', frameon=False, fontsize=5, 
                  handlelength=1.2, handletextpad=0.5, 
                  borderaxespad=0.5, labelspacing=0.3)
        
        # 添加浅色网格
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, linewidth=0.3)
        ax.set_axisbelow(True)

        # 保存图片
        plt.savefig(f'{output_name}.pdf', dpi=600)
        
        plt.show()

    def fgs_iden_and_ranks_dft():
        
        db_id_pro, db_sim_pro, db_smiles_pro, db_mw_pro = utils.load_dict('db_pro_info')
        # train and test data for dft and exp
        nist_plot_dict = utils.load_dict('unique_nist_plot_dict')
        nist_plot_dict_er = utils.load_dict('unique_nist_plot_dict_er')

        train_size = 2247
        train_size_er = 2138
        
        # matrix calculation
        # train_data = utils.load_dict('unique_train_data_no_Dsub')
        # test_data = utils.load_dict('unique_test_data_no_Dsub')
        # cleaned_train_data = utils.load_dict('unique_clean_train_data_no_error_and_iminol')
        # cleaned_test_data = utils.load_dict('unique_clean_test_data_no_error_and_iminol')
        
        # test_pred_ggnd = utils.load_dict('unique_ggnd_test_pred')
        # test_pred_ggnder = utils.load_dict('unique_ggnder_test_pred')
        test_pred_db = utils.load_dict('db_pro_pred')
        test_pred_db_er = utils.load_dict('db_pro_pred_er')
  
        # more memory need but less time
        quasi_iden_matrix = GPUCalculator().pearson_matrix_gpu(nist_plot_dict['exp_list'], np.concatenate((nist_plot_dict['pred_list'], test_pred_db)))
        dft_iden_matrix = GPUCalculator().pearson_matrix_gpu(nist_plot_dict['exp_list'], np.concatenate((nist_plot_dict['sim_list'], db_sim_pro)))
        
        # low memory but more time
        # quasi_mvs, quasi_drs, quasi_max_ids = iden_matrix_process_single(nist_plot_dict['exp_list'], 
        #                                                                   np.concatenate((nist_plot_dict['pred_list'], test_pred_db)),
        #                                                                   'unique_pearson_matrix_quasi')
        # dft_mvs, dft_drs, dft_max_ids = iden_matrix_process_single(nist_plot_dict['exp_list'], 
        #                                                             np.concatenate((nist_plot_dict['sim_list'], db_sim_pro)),
        #                                                             'unique_pearson_matrix_dft')
        # quasi_mvs_er, quasi_drs_er, quasi_max_ids_er = iden_matrix_process_single(nist_plot_dict_er['exp_list'], 
        #                                                                   np.concatenate((nist_plot_dict_er['pred_list'], test_pred_db_er)),
        #                                                                   'unique_pearson_matrix_quasi_er')
        # dft_mvs_er, dft_drs_er, dft_max_ids_er = iden_matrix_process_single(nist_plot_dict_er['exp_list'], 
        #                                                             np.concatenate((nist_plot_dict_er['sim_list'], db_sim_pro)),
        #                                                             'unique_pearson_matrix_dft_er')
        
        quasi_mvs, quasi_drs, quasi_max_ids = max_value_indices_and_sorting_order(utils.load_dict('unique_pearson_matrix_quasi'))
        dft_mvs, dft_drs, dft_max_ids = max_value_indices_and_sorting_order(utils.load_dict('unique_pearson_matrix_dft'))

        quasi_mvs_er, quasi_drs_er, quasi_max_ids_er = max_value_indices_and_sorting_order(utils.load_dict('unique_pearson_matrix_quasi_er'))
        dft_mvs_er, dft_drs_er, dft_max_ids_er = max_value_indices_and_sorting_order(utils.load_dict('unique_pearson_matrix_dft_er'))
        
        quasier_results4rank = {
            'max_values': quasi_mvs_er,
            'diag_ranks': quasi_drs_er,
            'max_ids': quasi_max_ids_er
            }
        
        dfter_results4rank = {
            'max_values': dft_mvs_er,
            'diag_ranks': dft_drs_er,
            'max_ids': dft_max_ids_er
            }
        
        # utils.save_dict(quasier_results4rank, 'quasier_results4rank')
        # utils.save_dict(dfter_results4rank, 'dfter_results4rank')

        # # all_id需要按照matrix计算顺序排序
        # fgs_ana_quasi = fgs_ana(nist_plot_dict['id_list'][train_size:], 
        #                         quasi_max_ids[train_size:], 
        #                         np.concatenate((nist_plot_dict['id_list'], db_id_pro)))
        # fgs_ana_dft = fgs_ana(nist_plot_dict['id_list'][train_size:], 
        #                       dft_max_ids[train_size:], 
        #                       np.concatenate((nist_plot_dict['id_list'],db_id_pro)))
        
        # fgs_ana_quasi_er = fgs_ana(nist_plot_dict_er['id_list'][train_size_er:], 
        #                             quasi_max_ids_er[train_size_er:], 
        #                             np.concatenate((nist_plot_dict_er['id_list'],db_id_pro)))
        
        # # utils.save_dict(fgs_ana_quasi_er, 'fgs_analysis_quasi_er')
        # fgs_ana_dft_er = fgs_ana(nist_plot_dict_er['id_list'][train_size_er:], 
        #                          dft_max_ids_er[train_size_er:], 
        #                          np.concatenate((nist_plot_dict_er['id_list'],db_id_pro)))
        
        
        # plot_natcomsci_performance([fgs_ana_quasi, fgs_ana_dft, fgs_ana_quasi_er, fgs_ana_dft_er])
        # quasier_results4rank = utils.load_dict('quasier_results4rank')
        # dfter_results4rank = utils.load_dict('dfter_results4rank')
        
        # fgs_ana_quasi_er = utils.load_dict('fgs_analysis_quasi_er')
        # print(fgs_ana_quasi_er)
        # fg_compare_heatmap(fgs_ana_quasi_er, model='All', save_name='fgs_compare')
        # plot_fgs_heatmap_el()
        
        # fg_compare_heatmap(fgs_ana_dft_er, model='All')
        # fg_compare_heatmap(fgs_ana_dft, model='All')
        # fg_compare_heatmap(fgs_ana_quasi, model='All')
        # fg_compare_heatmap(fgs_ana_quasi_er, model='EL')
        # plot_fgs_heatmap_el(nist_plot_dict_er, db_id_pro, db_smiles_pro, test_pred_db_er)
        
        # all_mw = np.concatenate((nist_plot_dict['mw_list'], db_mw_pro))  
        # all_mw_er = np.concatenate((nist_plot_dict_er['mw_list'], db_mw_pro)) 
        
        # dft_ms_ranks = detect_related_exp_ir_rank(nist_plot_dict['exp_list'], nist_plot_dict['mw_list'], 
        #                                           all_mw, 'unique_pearson_matrix_dft')
        # quasi_ms_ranks = detect_related_exp_ir_rank(nist_plot_dict['exp_list'], nist_plot_dict['mw_list'], 
        #                                             all_mw, 'unique_pearson_matrix_quasi') 

        # dft_ms_ranks_er = detect_related_exp_ir_rank(nist_plot_dict_er['exp_list'], nist_plot_dict_er['mw_list'], 
        #                                           all_mw_er, 'unique_pearson_matrix_dft_er')
        # quasi_ms_ranks_er = detect_related_exp_ir_rank(nist_plot_dict_er['exp_list'], nist_plot_dict_er['mw_list'], 
        #                                             all_mw_er, 'unique_pearson_matrix_quasi_er')     
        
        # utils.save_dict(dft_ms_ranks, 'dft_ms_ranks')
        # utils.save_dict(dft_ms_ranks_er, 'dft_ms_ranks_er')
        # utils.save_dict(quasi_ms_ranks, 'quasi_ms_ranks')
        # utils.save_dict(quasi_ms_ranks_er, 'quasi_ms_ranks_er')
            
        dft_ms_ranks=utils.load_dict('dft_ms_ranks')
        dft_ms_ranks_er=utils.load_dict('dft_ms_ranks_er')
        quasi_ms_ranks=utils.load_dict('quasi_ms_ranks')
        quasi_ms_ranks_er=utils.load_dict('quasi_ms_ranks_er')
        
        # suc_rate_rank_comparison([quasi_drs, dft_drs], ['Quasi_exp IR', 'DFT IR'], 21)
        # suc_rate_rank_comparison([quasi_ms_ranks, quasi_drs, dft_ms_ranks, dft_drs], ['Quasi_exp IR + MS','Quasi_exp IR', 'DFT IR + MS', 'DFT IR'], 21)
        suc_rate_rank_comparison([quasi_drs, dft_drs, quasi_drs_er, dft_drs_er], ['Quasi-exp IR','DFT IR','Quasi-exp/curated IR', 'DFT/curated IR'], 21)
        
        # quasier_results4rank = utils.load_dict('quasier_results4rank')
        # dfter_results4rank = utils.load_dict('dfter_results4rank')
        
        # suc_rate_rank_comparison([dft_ms_ranks_er, dfter_results4rank['diag_ranks'], quasi_ms_ranks_er, quasier_results4rank['diag_ranks']], ['DFT IR + MS','DFT IR','Quasi-exp IR + MS', 'Quasi-exp IR'], 21)

    def fgs_iden_and_ranks_xtb():
        
        def gen_db_pro_info_xtb():
            db_id_pro, db_sim_pro, db_smiles_pro, db_mw_pro = utils.load_dict('db_pro_info')
            db_xtb_data = data_loader.load_xtb_db_data()
            
            db_xtb_id_pro = []
            db_xtb_sim_pro = []
            db_xtb_smiles_pro = []
            db_xtb_mw_pro = []
            
            db_xtb_data_sqsim = [utils.sqrt_normalization(data) for data in db_xtb_data['xtb_ir_list']]
            
            xtb_db_id_sim_map = {}
        
            for idn, sim_ir in zip(db_xtb_data['xtb_id_list'], db_xtb_data_sqsim):
                xtb_db_id_sim_map[idn] = sim_ir
            
            for idn, smiles, mw in zip(db_id_pro, db_smiles_pro, db_mw_pro):
                if idn in db_xtb_data['xtb_id_list']:
                    db_xtb_id_pro.append(idn)
                    db_xtb_sim_pro.append(xtb_db_id_sim_map[idn])
                    db_xtb_smiles_pro.append(smiles)
                    db_xtb_mw_pro.append(mw)
            
            utils.save_dict((db_xtb_id_pro, db_xtb_sim_pro, db_xtb_smiles_pro, db_xtb_mw_pro), 'db_pro_info_xtb')

        def matrix_cal():
            
            db_xtb_id_pro, db_xtb_sim_pro, db_xtb_smiles_pro, db_xtb_mw_pro = utils.load_dict('db_pro_info_xtb')
            nist_plot_dict_xtb = utils.load_dict('unique_nistxtb_plot_dict')
            modelname_xtb_all = 'unique_xtb_train_all'
            xtb_pred_db = gpr_operations.predict_tf(utils.load_model(modelname_xtb_all), db_xtb_sim_pro)
            
            # low memory but more time
            augxtb_mvs, augxtb_drs, augxtb_max_ids = iden_matrix_process_single(nist_plot_dict_xtb['exp_list'], 
                                                                              np.concatenate((nist_plot_dict_xtb['pred_list'], xtb_pred_db)),
                                                                              'unique_pearson_matrix_augxtb')
            xtb_mvs, xtb_drs, xtb_max_ids = iden_matrix_process_single(nist_plot_dict_xtb['exp_list'], 
                                                                        np.concatenate((nist_plot_dict_xtb['sim_list'], db_xtb_sim_pro)),
                                                                        'unique_pearson_matrix_xtb')
        
        # gen_db_pro_info_xtb() 
        # matrix_cal()
        
        augxtb_mvs, augxtb_drs, augxtb_max_ids = max_value_indices_and_sorting_order(utils.load_dict('unique_pearson_matrix_augxtb'))
        xtb_mvs, xtb_drs, xtb_max_ids = max_value_indices_and_sorting_order(utils.load_dict('unique_pearson_matrix_xtb'))
        
        # FGs identification
        
        # train_size = 2247
        # train_size_er = 2138
        
        # fgs_ana_augxtb = fgs_ana(nist_plot_dict_xtb['id_list'][train_size:], 
        #                         augxtb_max_ids[train_size:], 
        #                         np.concatenate((nist_plot_dict_xtb['id_list'],db_xtb_id_pro)))
        
        # fgs_ana_xtb = fgs_ana(nist_plot_dict_xtb['id_list'][train_size:], 
        #                       xtb_max_ids[train_size:], 
        #                       np.concatenate((nist_plot_dict_xtb['id_list'],db_xtb_id_pro)))
        
        suc_rate_rank_comparison([augxtb_drs, xtb_drs], ['aug_xtb IR', 'xtb IR'], 21)
        
    def fgs_iden_and_ranks_xtbmd():
        nist_plot_dict_xtbmd = utils.load_dict('unique_nistxtbmd_plot_dict')

        aug_iden_matrix = GPUCalculator().pearson_matrix_gpu(nist_plot_dict_xtbmd['exp_list'], nist_plot_dict_xtbmd['pred_list'])
        xtbmd_iden_matrix = GPUCalculator().pearson_matrix_gpu(nist_plot_dict_xtbmd['exp_list'], nist_plot_dict_xtbmd['sim_list'])
        
        aug_mvs, aug_drs, aug_max_ids = max_value_indices_and_sorting_order(aug_iden_matrix)
        xtbmd_mvs, xtbmd_drs, xtbmd_max_ids = max_value_indices_and_sorting_order(xtbmd_iden_matrix)
        
        suc_rate_rank_comparison([aug_drs, xtbmd_drs], ['aug_xtb_md IR', 'xtb_md IR'], 21)
    
    def rank_case(id_, save_prefix="rank_case"):
        """
        显示前5个最相似分子的结构图和光谱图
        符合Nature期刊规范，保存为PDF
        """
        db_id_pro, db_sim_pro, db_smiles_pro, db_mw_pro = utils.load_dict('db_pro_info')
        nist_plot_dict_er = utils.load_dict('unique_nist_plot_dict_er')
        
        train_size = 2247
        train_size_er = 2138
        test_pred_db_er = utils.load_dict('db_pro_pred_er')
        rank_matrix = utils.load_dict('unique_pearson_matrix_quasi_er')
        all_id = np.concatenate((nist_plot_dict_er['id_list'], db_id_pro))
        
        try:
            id_index = np.where(all_id == id_)[0][0]
            print('id_index: ', id_index)
        except IndexError:
            raise ValueError(f"ID {id_} not found in all_id list")
        
        # 获取相似度向量
        similarity_row = rank_matrix[id_index]
        top_5_indices = np.argsort(similarity_row)[::-1][:5]  # 只取前5个
        top_5_ids = all_id[top_5_indices]
        top_5_similarities = similarity_row[top_5_indices]
        all_smiles = np.concatenate((nist_plot_dict_er['smiles_list'], db_smiles_pro))
        top_5_smiles = all_smiles[top_5_indices]
        
        # 1. 网格分子结构图 (4×2.5 cm)
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        # 创建分子结构网格图 (1行5列)
        width_cm1, height_cm1 = 4, 2.5
        width_inch1 = width_cm1 / 2.54
        height_inch1 = height_cm1 / 2.54
        
        fig_mol, axes_mol = plt.subplots(1, 5, figsize=(width_inch1, height_inch1), dpi=600)
        
        for i, (smiles, id_val, similarity) in enumerate(zip(top_5_smiles, top_5_ids, top_5_similarities)):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    img = Draw.MolToImage(mol, size=(100, 100))  # 减小图像尺寸
                    axes_mol[i].imshow(img)
                    # 添加标签（更小字体）
                    axes_mol[i].set_title(f"Rank {i+1}\n{similarity:.3f}", 
                                         fontsize=5, pad=3)
                    axes_mol[i].axis('off')
                else:
                    axes_mol[i].axis('off')
                    axes_mol[i].set_title(f"Rank {i+1}\nInvalid", fontsize=5, pad=3)
            except Exception as e:
                axes_mol[i].axis('off')
                axes_mol[i].set_title(f"Rank {i+1}\nError", fontsize=5, pad=3)
        
        plt.tight_layout()
        
        # 保存PDF
        save_path_mol = f"{save_prefix}_molecules.pdf"
        fig_mol.savefig(save_path_mol, format='pdf', dpi=600)
        print(f"已保存分子结构图: {save_path_mol} (尺寸: {width_cm1}×{height_cm1} cm)")
        
        plt.show()
        plt.close(fig_mol)
        
        # 2. 光谱对比图 (4×6 cm)
        all_spectra_pred = np.concatenate((nist_plot_dict_er['pred_list'], test_pred_db_er))
        wavenumber = utils.load_npy('wavenumber_550-3846-4')
        query_spectrum = all_spectra_pred[id_index]
        
        width_cm2, height_cm2 = 4, 6
        width_inch2 = width_cm2 / 2.54
        height_inch2 = height_cm2 / 2.54
        
        fig_spec, axes_spec = plt.subplots(5, 1, figsize=(width_inch2, height_inch2), 
                                          dpi=600, sharex=True)
        
        for i in range(5):  # 只显示5个
            idx = top_5_indices[i]
            spectrum = all_spectra_pred[idx]
            similarity = top_5_similarities[i]
            
            # 绘制光谱
            axes_spec[i].plot(wavenumber, query_spectrum, 
                             linewidth=0.6, color='black', label=f'Query')
            axes_spec[i].plot(wavenumber, spectrum, 
                             linewidth=0.8, color='#A51C36', alpha=0.8,
                             label=f'Rank {i+1}: {similarity:.3f}')
            axes_spec[i].set_yticklabels([]) 
            axes_spec[i].set_xticklabels([]) 
            axes_spec[i].tick_params(axis='y', length=0)
            axes_spec[i].tick_params(axis='x', length=0)
            # 设置坐标轴
            # if i == 4:
            #     axes_spec[i].set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=5)
            # axes_spec[i].set_ylabel('Intensity', fontsize=5)
            
            # 设置边框线宽
            for spine in axes_spec[i].spines.values():
                spine.set_linewidth(0.5)
            
            # 设置刻度（更小）
            # axes_spec[i].tick_params(axis='both', labelsize=5, width=0.5)
            
            # 添加图例（更小）
        #     axes_spec[i].legend(loc='upper right', fontsize=4.5, frameon=True,
        #                        edgecolor='black', facecolor='white', framealpha=0.9)
        #     axes_spec[i].get_legend().get_frame().set_linewidth(0.3)
            
        #     axes_spec[i].invert_xaxis()  # 红外光谱通常从高波数到低波数
        #     axes_spec[i].grid(True, alpha=0.2, linewidth=0.2, linestyle='--')
        
        # plt.tight_layout()
        
        # 保存PDF
        save_path_spec = f"{save_prefix}_spectra.pdf"
        fig_spec.savefig(save_path_spec, format='pdf', dpi=600)
        print(f"已保存光谱对比图: {save_path_spec} (尺寸: {width_cm2}×{height_cm2} cm)")
        
        plt.show()
        plt.close(fig_spec)
        
        # 3. 单独显示查询分子光谱 (4×2.5 cm)
        width_cm3, height_cm3 = 4, 2.5
        width_inch3 = width_cm3 / 2.54
        height_inch3 = height_cm3 / 2.54
        
        fig_query, ax_query = plt.subplots(figsize=(width_inch3, height_inch3), dpi=600)
        ax_query.plot(wavenumber, query_spectrum, linewidth=0.8, color='black')
        ax_query.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=5)
        ax_query.set_ylabel('Intensity', fontsize=5)
        ax_query.invert_xaxis()
        
        # 设置边框和刻度
        for spine in ax_query.spines.values():
            spine.set_linewidth(0.5)
        ax_query.tick_params(axis='both', labelsize=5, width=0.5)
        # ax_query.grid(True, alpha=0.2, linewidth=0.2, linestyle='--')
        
        # plt.tight_layout()
        
        # 保存PDF
        save_path_query = f"{save_prefix}_query_spectrum.pdf"
        fig_query.savefig(save_path_query, format='pdf', dpi=600)
        print(f"已保存查询分子光谱: {save_path_query} (尺寸: {width_cm3}×{height_cm3} cm)")
        
        plt.show()
        plt.close(fig_query)
        
        # 返回详细信息
        result = {
            'query_id': id_,
            'top_5_ids': top_5_ids.tolist(),
            'top_5_indices': top_5_indices.tolist(),
            'top_5_similarities': top_5_similarities.tolist(),
            'top_5_smiles': top_5_smiles.tolist()
        }
        
        print(result)
        return result
        # return result
    
    # fgs_iden_and_ranks_xtb()
    # fgs_iden_and_ranks_xtbmd()
    fgs_iden_and_ranks_dft()
    # rank_case(402346)
    
def plot_largemol():
    set_natcomsci_style()
    colors_set = ['#7ABBDB', '#A51C36', '#84BA42', '#682487','#DBB428']
    
    def plot_two_ir(ir1, ir2, label1, label2):
        """
        绘制两个IR光谱在同一图中，并显示它们的差值
        
        参数:
            ir1: 第一个IR光谱数据
            ir2: 第二个IR光谱数据
            label1: 第一个光谱的标签
            label2: 第二个光谱的标签
            idn: 图表标题
        """
        # 颜色设置
        color_ir1 = '#A51C36'  # 红色
        color_ir2 = '#7ABBDB'  # 蓝色
        color_diff = 'purple'  # 绿色
        color_ref_line1 = 'green'
        color_ref_line2 = 'orange'
        color_ref_line3 = 'red'
        
        # 加载wavenumber数据
        x = utils.load_npy('wavenumber_550-3846-4')
        
        # 计算差值
        diff = np.array(ir1) - np.array(ir2)
        
        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=600, 
                                gridspec_kw={'height_ratios': [1, 1]})
        
        # 第一个子图：两个IR光谱
        ax1 = axes[0]
        ax1.plot(x, ir1, label=label1, color=color_ir1, linewidth=1.5)
        ax1.plot(x, ir2, label=label2, color=color_ir2, linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)', fontweight='bold')
        ax1.set_ylabel('Intensity', fontweight='bold')
        # ax1.set_title(idn, fontweight='bold')
        ax1.legend(prop={'size': 8}, loc='best')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # 反转x轴（IR光谱通常从高波数到低波数）
        ax1.set_xlim(x.max(), x.min())
        
        # 第二个子图：差值图
        ax2 = axes[1]
        ax2.plot(x, diff, label=f'Difference ({label1} - {label2})', 
                 color=color_diff, linewidth=1.2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)', fontweight='bold')
        ax2.set_ylabel('Δ Intensity', fontweight='bold')
        ax2.legend(prop={'size': 7}, loc='best')
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        ax2.axhline(y=-0.1, color=color_ref_line1, linestyle='--', linewidth=0.8, alpha=0.6)
        ax2.axhline(y=-0.3, color=color_ref_line2, linestyle='--', linewidth=0.8, alpha=0.6)
        ax2.axhline(y=-0.5, color=color_ref_line3, linestyle='--', linewidth=0.8, alpha=0.6)
        
        # 反转x轴（与上面的图保持一致）
        ax2.set_xlim(x.max(), x.min())
        ax2.set_ylim(-1, 1)
        
        # 调整布局并显示
        plt.tight_layout()
        plt.show()
        
    def plot_four_ir_comparison(ir_base, ir_compare_list, label_base, label_compare_list,
                           width_cm=8, height_cm=10, save_name="four_ir_comparison"):
        """
        绘制基准IR光谱与三个对比光谱在同一张图中
        上面：四个光谱对比，下面：三个差值
        """
        # 颜色设置
        color_base = 'black'  # 基准为黑色
        colors_compare = ['#A51C36', '#7ABBDB', '#84BA42']  # 三个对比颜色：红、蓝、绿
        colors_diff = ['#682487', '#DBB428', '#FF6B6B']  # 差值颜色
        
        # 加载wavenumber数据
        x = utils.load_npy('wavenumber_550-3846-4')
        
        # 创建上下两个子图
        width_inch = width_cm / 2.54
        height_inch = height_cm / 2.54
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width_inch, height_inch), 
                                      dpi=600, sharex=True)
        
        # 顶部子图：四个光谱对比
        ax1.plot(x, ir_base, label=label_base, 
                color=color_base, linewidth=0.5)
        
        for i, (ir_compare, label_compare) in enumerate(zip(ir_compare_list, label_compare_list)):
            if i >= 3:  # 只处理前三个
                break
            ax1.plot(x, ir_compare, label=label_compare, 
                    color=colors_compare[i], linewidth=0.5, alpha=0.6)
        
        ax1.set_ylabel('Intensity', fontsize=6)
        
        # 添加图例（有边框和底色，左上角）
        legend1 = ax1.legend(loc='upper left',
                            fontsize=6,
                            frameon=True,
                            fancybox=False,
                            edgecolor='black',
                            facecolor='white',
                            framealpha=0.9,
                            borderpad=0.6)
        legend1.get_frame().set_linewidth(0.5)
        
        ax1.grid(True, alpha=0.2, linewidth=0.3, linestyle='--')
        ax1.invert_xaxis()  # 反转x轴
        
        # 设置边框和刻度
        for spine in ax1.spines.values():
            spine.set_linewidth(0.5)
        ax1.tick_params(axis='both', labelsize=6, width=0.5)
        
        # 底部子图：三个差值
        for i, (ir_compare, label_compare) in enumerate(zip(ir_compare_list, label_compare_list)):
            if i >= 3:  # 只处理前三个
                break
            diff = np.array(ir_base) - np.array(ir_compare)
            
            # 创建掩码：仅显示负值部分
            mask = diff < 0
            masked_x = x[mask]
            masked_diff = diff[mask]
            
            # 使用原图颜色（colors_compare）
            ax2.plot(masked_x, masked_diff, 
                    label=f'{label_base} - {label_compare}',
                    color=colors_compare[i], linewidth=0.5)
            
            print(np.min(masked_diff))
        
        ax2.axhline(y=0, color='black', linestyle='-', 
                   linewidth=0.5, alpha=0.5)
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=6)
        ax2.set_ylabel('Δ Intensity (Negative)', fontsize=6)
        
        # 设置y轴范围（仅显示负值部分）
        # 可以自动计算负值范围或手动设置
        if len(ir_compare_list) > 0:
            y_min = min([min(np.array(ir_base) - np.array(ir_compare)) 
                        for ir_compare in ir_compare_list[:3]])
            ax2.set_ylim(y_min * 1.1, 0.1)  # 给底部留一点空间，顶部到0
        
        # 设置边框和刻度
        for spine in ax2.spines.values():
            spine.set_linewidth(0.5)
        ax2.tick_params(axis='both', labelsize=6, width=0.5)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存PDF
        save_path = f"{save_name}.pdf"
        fig.savefig(save_path, format='pdf', dpi=600)
        print(f"已保存: {save_path}")
        
        plt.show()
        plt.close(fig)
    def plot_ir_frag(ir_list, label_list, i=0):
        x = utils.load_npy('wavenumber_550-3846-4')
        plt.figure(figsize=(12, 4), dpi=600)
        for ir, label in zip(ir_list, label_list):
            plt.plot(x, ir, label=label, color=colors_set[i])
            i += 1
        plt.xlabel('Wavenumber (cm$^{-1}$)', fontweight='bold')
        plt.ylabel('Intensity', fontweight='bold')
        plt.legend(prop={'size': 10}, fontsize = 'small', )
        plt.show()
        
    def lorentzian(x,peak,height,width):
        a = width**2./4.
        return float(height)*a/( (peak-x)**2 + a )

    class Spectrum(object):
        def __init__(self,start,end,numpts,peaks,width,formula):
            self.start = start
            self.end = end
            self.numpts = numpts
            self.peaks = peaks
            self.width = width
            self.formula = formula

            # len(peaks) is the number of spectra in this object
            self.spectrum = np.zeros( (numpts,len(peaks)),"d")
            self.xvalues = np.arange(numpts)*float(end-start)/(numpts-1) + start
            for i in range(numpts):
                x = self.xvalues[i]
                for spectrumno in range(len(peaks)):
                    for (pos,height) in peaks[spectrumno]:
                        self.spectrum[i,spectrumno] = self.spectrum[i,spectrumno] + formula(x,pos,height,width)

    def ir_spectrum(freq, act):
        peaks = [[(x,y) for x, y in zip(freq, act)]]
        start = 546
        end = 3846
        numpts = 825
        width = 24
        spectrum = Spectrum(start,end,numpts,peaks,width,lorentzian)
        for x in range(0,numpts):
            if spectrum.spectrum[x,0]<1e-20:
                spectrum.spectrum[x,0] = 0.

        wavenumber = spectrum.xvalues
        intensity = spectrum.spectrum[:,0]
        sq_Int = utils.sqrt_normalization(intensity)
        return wavenumber, intensity, sq_Int
    
    def inverse_sqrt_norm(sqrt_norm_data):
        """
        将平方根归一化数据转换为普通归一化数据
        
        参数:
            sqrt_norm_data: 经过 sqrt_norm() 处理的数据
            
        返回:
            普通线性归一化后的数据
        """
        arr = np.array(sqrt_norm_data)
        
        # 平方运算去除平方根
        squared = arr ** 2
        
        return squared.tolist()
        
    def large_mol_compare():
        largemol_dict = utils.load_dict('large_mol_dft_dict')
        largemol_exp = utils.load_npy('large_mol_exp_ir')
        
        # largemol_dict = utils.load_dict('large_mol_dft_dict_2')
        # largemol_exp = utils.load_npy('large_mol_exp_ir_2')
        
        # not in match
        sim_list = utils.load_npy('best_sq_sim_list')
        exp_list = utils.load_npy('best_sq_fit_list') 
        id_list = utils.load_npy('best_id_list')
        
        re_exp_list = []
        re_id_list = []
        re_sim_list = []
        id_c9h12o = [402686, 401572]
        for i, idn in enumerate(id_list):
            if idn in id_c9h12o:
                re_sim_list.append(sim_list[i])
                re_exp_list.append(exp_list[i])
                re_id_list.append(idn)
        
        scale_factor = 0.965
        frag_ir_list = []
        
        for ir_info in largemol_dict:
            freq = [float(f) for f in ir_info['Freq'] if f is not None and f.strip() != '']
            act = [float(f) for f in ir_info['IR_act'] if f is not None and f.strip() != '']
            freq_sf = [f * scale_factor for f in freq]
            wavenumber, intensity, sq_Int = ir_spectrum(freq_sf, act)
            frag_ir_list.append(sq_Int)
        
        model_name = 'unique_ggnder_train_all'
        gpr_tf = utils.load_model(model_name)
        frag_pred = gpr_operations.predict_tf(gpr_tf, frag_ir_list) 
        
        frag_notin = gpr_operations.predict_tf(gpr_tf, re_sim_list)
        
        # frag_pred = [normalization(data) for data in frag_pred_list]
        # r_match_in = utils.pearson(largemol_exp, frag_ir_list[2])
        # r_match_not1 = utils.pearson(largemol_exp, frag_notin[0])
        # r_match_not2 = utils.pearson(largemol_exp, frag_notin[1])
        # print(r_match_in, r_match_not1, r_match_not2)
        
        largemol_expn = inverse_sqrt_norm(largemol_exp)
        frag_inn = inverse_sqrt_norm(frag_pred[2])
        frag_notin_1n = inverse_sqrt_norm(frag_notin[0])
        frag_notin_2n = inverse_sqrt_norm(frag_notin[1])
        
        frag_inn_sim = inverse_sqrt_norm(frag_ir_list[2])
        frag_notin_1n_sim = inverse_sqrt_norm(re_sim_list[0])
        frag_notin_2n_sim = inverse_sqrt_norm(re_sim_list[1])
        
        frag_dis_in = [ (y - x) if (y - x) < 0 else 0 for x, y in zip(frag_inn, largemol_expn) ]
        frag_dis_notin_1 = [(y - x) if (y - x) < 0 else 0 for x,y in zip(frag_notin_1n, largemol_expn)]
        frag_dis_notin_2 = [(y - x) if (y - x) < 0 else 0 for x,y in zip(frag_notin_2n, largemol_expn)]
        
        # plot_ir_frag([frag_dis_in, frag_dis_notin_1, frag_dis_notin_2], ['Frag in mol', 'Frag1 not in mol ', 'Frag2 not in mol'])
        
        # plot_two_ir(largemol_expn, frag_inn, 'Mol', 'Frag in Mol')
        # plot_two_ir(largemol_expn, frag_notin_1n, 'Mol', 'Frag not in Mol')
        # plot_two_ir(largemol_expn, frag_notin_2n, 'Mol', 'Frag not in Mol')
        
        plot_four_ir_comparison(largemol_expn, [frag_inn, frag_notin_1n, frag_notin_2n,], 
                                'LargeMol', ['Frag1 in Mol', 'Frag2 not in Mol', 'Frag3 not in Mol'])
        
        # plot_two_ir(largemol_expn, frag_inn_sim, 'Mol', 'Frag in Mol')
        # plot_two_ir(largemol_expn, frag_notin_1n_sim, 'Mol', 'Frag not in Mol')
        # plot_two_ir(largemol_expn, frag_notin_2n_sim, 'Mol', 'Frag not in Mol')
        
        # plot_ir_frag([largemol_exp],['Exp'])
        # # plot_ir_frag([[x+y for x, y in zip(frag_pred[2], frag_pred[1])]], ['Aug_Frag1 + Aug_Frag2'], i=2)
        # plot_ir_frag([largemol_exp, frag_pred[2], frag_pred[1]], ['Exp','Aug_Frag1','Aug_Frag2'])
        # plot_ir_frag([largemol_exp, frag_ir_list[2], frag_ir_list[1]], ['Exp','Frag1','Frag2'])
        
        # plot_ir_frag([largemol_exp, frag_pred[0]], ['Exp','Quasi'])
        # plot_ir_frag([largemol_exp, frag_ir_list[0]], ['Exp','DFT'])
        # # frag_notin_exp = np.concatenate(([largemol_exp], frag_notin))
        # # plot_ir_frag(frag_notin_exp, ['Exp', 'notin frag 1', 'notin frag 2', 'notin frag 3'])
        
        # r_match_dft = utils.pearson(largemol_exp, [x+y for x, y in zip(frag_pred[2], frag_pred[1])])
        # r_match_quasi = utils.pearson(largemol_exp, [x+y for x, y in zip(frag_ir_list[2], frag_ir_list[1])])
        # r_match_dft_all = utils.pearson(largemol_exp, frag_ir_list[0])
        # r_match_quasi_all = utils.pearson(largemol_exp, frag_pred[0])
        
        # print(r_match_dft, r_match_dft_all, r_match_quasi, r_match_quasi_all)

    large_mol_compare() 

class SpectrumPlotter:
    def __init__(self, dpi=600):
        """Initialize the plotter with default settings"""
        self.dpi = dpi
        self.color_set = ['#7ABBDB', '#84BA42', '#A51C36']  # Default color palette
        
    def plot_histograms(self, similarity_lists, xlabel='Similarity Measure', titles=None):
        """
        Plot multiple histograms of similarity measures in a vertical stack
        
        Args:
            similarity_lists: List of lists containing similarity measures
            xlabel: Label for x-axis
            titles: Optional list of titles for each subplot
        """
        max_index = len(similarity_lists) - 1
        range_min, range_max = 0.0, 1.0
        num_bins = 100
        
        fig, axs = plt.subplots(len(similarity_lists), 1, sharex=True, dpi=self.dpi)
        
        if len(similarity_lists) == 1:
            axs = [axs]  # Ensure axs is always iterable
            
        for i, sl in enumerate(similarity_lists):
            mean_val = np.mean(sl)
            print(f"Mean similarity for dataset {i+1}: {mean_val:.4f}")
            
            hist, bins = np.histogram(sl, bins=num_bins, range=(range_min, range_max))
            axs[i].hist(bins[:-1], bins=bins, weights=hist, 
                        color=self.color_set[i % len(self.color_set)], 
                        edgecolor='w')
            
            if titles and i < len(titles):
                axs[i].set_title(titles[i])
        
        axs[max_index].set_xlim(range_min, range_max)
        axs[max_index].set_xlabel(xlabel, fontweight='bold')
        fig.text(0.04, 0.5, '#Cases', va='center', rotation='vertical', fontweight='bold')
        plt.tight_layout(rect=[0.05, 0, 1, 1]) 
        plt.show()
    
    def plot_roc_curve(self, similarity_matrix, model_name='Model', plot=True):
        """
        Calculate and optionally plot ROC curve for a similarity matrix
        
        Args:
            similarity_matrix: 2D numpy array of similarity scores
            model_name: Name of the model for labeling
            plot: Whether to display the plot
            
        Returns:
            Dictionary containing ROC metrics (fpr, tpr, thresholds, auc)
        """
        if not isinstance(similarity_matrix, np.ndarray):
            similarity_matrix = np.array(similarity_matrix)
            
        if similarity_matrix.ndim != 2:
            raise ValueError("Similarity matrix must be 2-dimensional")
        
        predicted_scores = []
        true_labels = []
        
        for r_idx, c_idx in np.ndindex(similarity_matrix.shape):
            value = similarity_matrix[r_idx, c_idx] 
            true_labels.append(1 if r_idx == c_idx else 0)
            predicted_scores.append(value)
        
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        if plot:
            self._plot_single_roc(fpr, tpr, roc_auc, model_name)
            
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc,
            'model': model_name
        }
    
    def plot_multiple_roc_curves(self, matrix_list, labels=None, title_suffix='', markers=None, colors=None):
        """
        Plot multiple ROC curves on the same axes for comparison
        
        Args:
            matrix_list: List of similarity matrices
            labels: List of labels for each model
            title_suffix: Additional text for plot title
            markers: List of markers for each curve
            colors: List of colors for each curve
        """
        if labels is None:
            labels = [f'Model {i+1}' for i in range(len(matrix_list))]
        elif len(labels) != len(matrix_list):
            raise ValueError("Number of labels must match number of matrices")
        
        # Set default markers and colors if not provided
        if markers is None:
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(matrix_list)]
        elif len(markers) != len(matrix_list):
            raise ValueError("Number of markers must match number of matrices")
        
        if colors is None:
            colors = plt.cm.tab10(range(len(matrix_list)))
        elif len(colors) != len(matrix_list):
            raise ValueError("Number of colors must match number of matrices")
        
        plt.figure(figsize=(10, 6), dpi=self.dpi)
        
        # Store results for potential further analysis
        roc_results = []
        
        # ls_list = [':', '-', ':', '-']
        # ls_list2 = ['--', '-', '--', '-']
        # lw_list = [3,2,3,2]
        # colors_new = ['#FF0000', '#FF0000', '#0066CC','#0066CC']
        for i, matrix in enumerate(matrix_list):
            result = self.plot_roc_curve(matrix, labels[i], plot=False)
            roc_results.append(result)
            
            # Plot with marker and color
            color = colors[i] if isinstance(colors[i], str) else colors[i]
            # marker = markers[i]
            
            # Use marker every 10th point to avoid overcrowding
            plt.plot(result['fpr'], result['tpr'], 
                    color=color, 
                    # linestyle=ls_list[i],
                    # lw = lw_list[i],
                    lw=2,
                    # marker=marker,
                    # markevery=0.1,  # Place marker every 10% of the curve
                    # markersize=6,
                    label=f"{labels[i]} (AUC={result['auc']:.3f})")
        
        # Reference line
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        
        title = f'ROC Curve Comparison{title_suffix}'
        # plt.title(title, fontsize=14)
        plt.legend(loc="lower right", fontsize=16)
        plt.grid(True)
        plt.show()
        
        return roc_results
    
    def _plot_single_roc(self, fpr, tpr, roc_auc, model_name):
        """Internal method to plot a single ROC curve"""
        plt.figure(figsize=(10, 6), dpi=self.dpi)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})\nModel: {model_name}')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title(f'ROC Curve: {model_name}', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.show()

class SpectrumAnalysis:
    def __init__(self, plotter=None):
        """Initialize with a plotter instance"""
        self.plotter = plotter if plotter else SpectrumPlotter()
        self.utils = utils  
        self.gpu_calculator = GPUCalculator()  
    
    def analyze_dft_data(self):
        """Analyze and plot DFT comparison data"""
        # Load data
        nist_plot_dict = self.utils.load_dict('unique_nist_plot_dict')
        nist_plot_dict_er = self.utils.load_dict('unique_nist_plot_dict_er')
        nist_plot_dict_nn = self.utils.load_dict('unique_nist_plot_dict_nn')
        
        # Calculate similarity lists
        sl_dft = [self.utils.pearson(u, v) for u, v in zip(nist_plot_dict['sim_list'], nist_plot_dict['exp_list'])]
        sl_gpr = [self.utils.pearson(u, v) for u, v in zip(nist_plot_dict['pred_list'], nist_plot_dict['exp_list'])]
        sl_nn = [self.utils.pearson(u, v) for u, v in zip(nist_plot_dict_nn['pred_list'], nist_plot_dict_nn['exp_list'])]
        
        # Plot histograms
        self.plotter.plot_histograms([sl_dft, sl_nn, sl_gpr], titles=['DFT', 'Neural Network', 'GPR'])
        
        # Calculate similarity matrices
        iden_matrix_quasi = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict['pred_list'], nist_plot_dict['exp_list'])
        iden_matrix_dft = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict['sim_list'], nist_plot_dict['exp_list'])
        iden_matrix_nn = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_nn['pred_list'], nist_plot_dict_nn['exp_list'])
        
        # Plot ROC comparisons
        self.plotter.plot_multiple_roc_curves([iden_matrix_quasi, iden_matrix_nn, iden_matrix_dft], 
                                            ['GPR', 'NN', 'DFT'])
        self.plotter.plot_multiple_roc_curves([iden_matrix_quasi, iden_matrix_dft], 
                                            ['GPR', 'DFT'])
        
        # Analyze error-curated data
        iden_matrix_quasi_er = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_er['pred_list'], nist_plot_dict_er['exp_list'])
        iden_matrix_dft_er = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_er['sim_list'], nist_plot_dict_er['exp_list'])
        self.plotter.plot_multiple_roc_curves([iden_matrix_quasi_er, iden_matrix_dft_er], 
                                            ['GPR (curated)', 'DFT (curated)'], 
                                            title_suffix=' - Curated Dataset')
    
    def analyze_xtb_data(self, md_data=False):
        """Analyze and plot xTB comparison data"""
        dict_key = 'unique_nistxtbmd_plot_dict' if md_data else 'unique_nistxtb_plot_dict'
        nist_plot_dict_xtb = self.utils.load_dict(dict_key)
        
        # Calculate similarity lists
        sl_xtb = [self.utils.pearson(u, v) for u, v in zip(nist_plot_dict_xtb['sim_list'], nist_plot_dict_xtb['exp_list'])]
        sl_augxtb = [self.utils.pearson(u, v) for u, v in zip(nist_plot_dict_xtb['pred_list'], nist_plot_dict_xtb['exp_list'])]
        
        # Plot histograms
        model_type = 'xTB-MD' if md_data else 'xTB'
        self.plotter.plot_histograms([sl_xtb, sl_augxtb], titles=[model_type, f'Aug-{model_type}'])
        
        # Calculate similarity matrices
        iden_matrix_xtb = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_xtb['sim_list'], nist_plot_dict_xtb['exp_list'])
        iden_matrix_augxtb = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_xtb['pred_list'], nist_plot_dict_xtb['exp_list'])
        
        # Plot ROC comparison
        self.plotter.plot_multiple_roc_curves([iden_matrix_xtb, iden_matrix_augxtb], 
                                             [model_type.lower(), f'aug-{model_type.lower()}'])
        
    def compare_dft_xtb_roc(self, include_md=False):
        """Compare DFT and xTB ROC curves in a single plot
        
        Args:
            include_md (bool): Whether to include xTB-MD data
        """
        # Load DFT data (include both DFT and quasi-experimental)
        nist_plot_dict = self.utils.load_dict('unique_nist_plot_dict')
        iden_matrix_dft = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict['sim_list'], nist_plot_dict['exp_list'])
        iden_matrix_quasi = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict['pred_list'], nist_plot_dict['exp_list'])
        
        # Load xTB data (include both xTB and aug-xTB)
        dict_key_xtb = 'unique_nistxtbmd_plot_dict' if include_md else 'unique_nistxtb_plot_dict'
        nist_plot_dict_xtb = self.utils.load_dict(dict_key_xtb)
        iden_matrix_xtb = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_xtb['sim_list'], nist_plot_dict_xtb['exp_list'])
        iden_matrix_augxtb = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_xtb['pred_list'], nist_plot_dict_xtb['exp_list'])
        
        # Prepare data for combined plot - ordered by importance (high to low)
        similarity_matrices = [iden_matrix_quasi, iden_matrix_augxtb, iden_matrix_dft, iden_matrix_xtb]
        labels = [
            'Quasi-Exp',
            'Aug-xTB-MD' if include_md else 'Aug-xTB',
            'DFT', 
            'xTB-MD' if include_md else 'xTB'
        ]
        
        # Define markers and colors for better visibility
        markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
        colors = ['#FF0000', '#FF6600', '#0066CC', '#666666']  # red, orange, blue, gray
        
        # Plot combined ROC curves with custom styling
        self.plotter.plot_multiple_roc_curves(
            similarity_matrices, 
            labels,
            title_suffix='',
            markers=markers,
            colors=colors
        )
    
    def compare_xtb_roc(self, include_md=True):
        """Compare xTB-MD and xTB ROC curves in a single plot
        
        Args:
            include_md (bool): Whether to include xTB-MD data
        """
        
        # Load xTB data (include both xTB and aug-xTB)
        dict_key_xtb = 'unique_nistxtb_plot_dict'
        dict_key_xtb_md = 'unique_nistxtbmd_plot_dict'
        nist_plot_dict_xtb = self.utils.load_dict(dict_key_xtb)
        nist_plot_dict_xtb_md = self.utils.load_dict(dict_key_xtb_md)
        
        iden_matrix_xtb = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_xtb['sim_list'], nist_plot_dict_xtb['exp_list'])
        iden_matrix_augxtb = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_xtb['pred_list'], nist_plot_dict_xtb['exp_list'])
        
        iden_matrix_xtb_md = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_xtb_md['sim_list'], nist_plot_dict_xtb_md['exp_list'])
        iden_matrix_augxtb_md = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_xtb_md['pred_list'], nist_plot_dict_xtb_md['exp_list'])
        
        # Prepare data for combined plot - ordered by importance (high to low)
        similarity_matrices = [iden_matrix_augxtb, iden_matrix_xtb, iden_matrix_augxtb_md, iden_matrix_xtb_md]
        labels = [
            'Aug-xTB',
            'xTB',
            'Aug-xTB-MD',
            'xTB-MD'
        ]
        
        # Define markers and colors for better visibility
        # markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
        colors = ['#FF0000', '#0066CC', '#FF6600', '#666666']  # red, orange, blue, gray
        
        # Plot combined ROC curves with custom styling
        self.plotter.plot_multiple_roc_curves(
            similarity_matrices, 
            labels,
            title_suffix='',
            colors=colors
        )
    
    def compare_all_methods_roc(self):
        """Compare all methods (DFT, xTB, augmented methods) in one ROC plot"""
        # Load all datasets
        nist_plot_dict = self.utils.load_dict('unique_nist_plot_dict')
        nist_plot_dict_xtb = self.utils.load_dict('unique_nistxtb_plot_dict')
        nist_plot_dict_nn = self.utils.load_dict('unique_nist_plot_dict_nn')
        
        # Calculate similarity matrices for all methods
        iden_matrix_dft = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict['sim_list'], nist_plot_dict['exp_list'])
        iden_matrix_gpr = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict['pred_list'], nist_plot_dict['exp_list'])
        iden_matrix_nn = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_nn['pred_list'], nist_plot_dict_nn['exp_list'])
        iden_matrix_xtb = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_xtb['sim_list'], nist_plot_dict_xtb['exp_list'])
        iden_matrix_augxtb = self.gpu_calculator.pearson_matrix_gpu(nist_plot_dict_xtb['pred_list'], nist_plot_dict_xtb['exp_list'])
        
        # Prepare data for comprehensive comparison
        similarity_matrices = [
            iden_matrix_dft,
            iden_matrix_xtb, 
            iden_matrix_gpr,
            iden_matrix_nn,
            iden_matrix_augxtb
        ]
        
        labels = [
            'DFT',
            'xTB',
            'GPR',
            'Neural Network', 
            'Aug-xTB'
        ]
        
        # Plot comprehensive ROC comparison
        self.plotter.plot_multiple_roc_curves(
            similarity_matrices,
            labels,
            title_suffix=' - Comprehensive Method Comparison'
        )
        
def plot_roc_method():
    set_natcomsci_style()
    def plot_roc_by_similarity(iden_matrix, indices, ai_model='Quasi_exp', plot_s='yes'):
        
        iden_matrix = np.array(iden_matrix)
        
        predicted_scores = []
        true_labels = []
        
        for i in range(len(iden_matrix)):
            row = iden_matrix[i]
            c_idx = indices[i]
            
            pcc_positive = row[i]
            predicted_scores.append(pcc_positive)
            true_labels.append(1)
            
            for c_i in c_idx:
                value = row[c_i]
                pcc_negative = value
                predicted_scores.append(pcc_negative)
                true_labels.append(0)
        
        true_labels_np = np.array(true_labels)
        predicted_scores_np = np.array(predicted_scores)
        
        fpr, tpr, thresholds = roc_curve(true_labels_np, predicted_scores_np, pos_label=1)
        roc_auc = auc(fpr, tpr)
        if plot_s == 'yes':
            print(f"AUC值: {roc_auc:.4f}")
            roc_plot(fpr, tpr, thresholds, roc_auc, ai_model)
        return roc_auc
        
    def roc_plot(fpr, tpr, thresholds, roc_auc, ai_model='Quasi_exp'): 

        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})\n(AI: {ai_model}, Sim: PCC, Neg: Subset)') # 修改图例
        plt.plot(fpr, tpr, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title(f'ROC Curve: AI ({ai_model}) vs. Experimental Spectra (Subset)', fontsize=14) # 修改标题
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.show()
        
    def extract_top_elements(matrix, k=500, ex='max'):
        """
        提取矩阵中（除对角线外）每一行最大或最小的k个值的索引和值
        
        参数:
        matrix: 输入矩阵（二维numpy数组）
        k: 要提取的元素数量（默认为500）
        ex: 'max'表示提取最大值，'min'表示提取最小值（默认为'max'）
        
        返回:
        (indices, values): 一个元组，包含两个二维numpy数组
            - indices: 每行包含对应行的top k个值的索引（不包括对角线）
            - values: 每行包含对应行的top k个值
        """
        # 创建一个副本以避免修改原始矩阵
        modified_matrix = matrix.copy()
        
        # 将对角线元素设置为无穷大或无穷小，确保它们不会被选中
        np.fill_diagonal(modified_matrix, -np.inf if ex == 'max' else np.inf)
        
        # 根据ex参数选择提取最大还是最小值
        if ex == 'max':
            # 获取每行最大的k个值的索引
            sorted_indices = np.argsort(-modified_matrix, axis=1)[:, :k]
        elif ex == 'min':
            # 获取每行最小的k个值的索引
            sorted_indices = np.argsort(modified_matrix, axis=1)[:, :k]
        else:
            raise ValueError("参数ex必须是'max'或'min'")
        
        # 使用高级索引提取对应的值
        rows = np.arange(matrix.shape[0]).reshape(-1, 1)  # 行索引
        values = matrix[rows, sorted_indices]  # 提取值
        
        return sorted_indices, values
    
    def sort_data_by_quasi(data):
        """
        按照quasi的值降序排列整个data字典
        
        参数:
            data (dict): 包含各种数据的字典，必须有'quasi'键
            
        返回:
            dict: 所有数组都按照quasi降序排列的新字典
        """
        # 获取quasi的降序排列索引
        sort_indices = np.argsort(data['quasi'])[::-1]
        
        # 创建排序后的新字典
        sorted_data = {}
        
        # 对每个键对应的数组按照索引重新排序
        for key in data:
            sorted_data[key] = [data[key][i] for i in sort_indices]
        
        return sorted_data
    
    def plot_combined_roc(Data, width_cm=6, height_cm=4, save_name="roc_combined"):
        set_natcomsci_style()
        
        # 创建图形（按Nature规范）
        width_inch = width_cm / 2.54
        height_inch = height_cm / 2.54
        fig, ax = plt.subplots(figsize=(width_inch, height_inch), dpi=600)
        
        # 绘制数据点
        ax.plot(Data['quasi'], Data['roc_quasi'], 'o-', 
                markersize=2, label='Quasi-exp', 
                color='#A51C36', linewidth=0.5, markeredgecolor='none')
        ax.plot(Data['dft'], Data['roc_dft'], 's--', 
                markersize=2, label='DFT', 
                color='#A51C36', linewidth=0.5, markeredgecolor='none')
        ax.plot(Data['augxtb'], Data['roc_augxtb'], 'o-', 
                markersize=2, label='Aug-xTB', 
                color='#7ABBDB', linewidth=0.5, markeredgecolor='none')
        ax.plot(Data['xtb'], Data['roc_xtb'], 's--', 
                markersize=2, label='xTB', 
                color='#7ABBDB', linewidth=0.5, markeredgecolor='none')
        
        # 设置坐标轴标签
        ax.set_xlabel('Average Maximum Similarity', fontsize=6)
        ax.set_ylabel('AUC Score', fontsize=6)
        
        # 设置刻度和范围
        ax.set_xlim(left=0.4)
        ax.set_xticks(np.arange(0.4, 1.0, 0.1))
        ax.tick_params(axis='both', labelsize=6, width=0.5)
        
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        # 添加图例（有边框和底色，位置可选）
        legend = ax.legend(loc='best',
                          fontsize=5,
                          frameon=True,
                          fancybox=False,
                          edgecolor='black',
                          facecolor='white',
                          framealpha=0.9,
                          borderpad=0.6,
                          handlelength=1.5,
                          handletextpad=0.5,
                          borderaxespad=0.5)
        legend.get_frame().set_linewidth(0.5)
        
        # 网格
        ax.grid(True, alpha=0.2, linewidth=0.3, linestyle='--')
        
        # 保存PDF
        save_path = f"{save_name}.pdf"
        fig.savefig(save_path, format='pdf', dpi=600)
        print(f"已保存: {save_path}")
        
        plt.show()
        plt.close(fig)
        
        # plt.plot(Data['quasi'], Data['roc_quasi'], 'b-o', markersize=4, label='quasi_exp')
        # plt.plot(Data['dft'], Data['roc_dft'], 'b--s', markersize=4, label='DFT')
        
        # plt.plot(Data['augxtb'], Data['roc_augxtb'], 'g-o', markersize=4, label='aug_xtb')
        # plt.plot(Data['xtb'], Data['roc_xtb'], 'g--s', markersize=4, label='xtb')
     
        # plt.plot(Data['augpcff'], Data['roc_augpcff'], 'r-o', markersize=4, label='aug_pcff')
        # plt.plot(Data['pcff'], Data['roc_pcff'], 'r--s', markersize=4, label='pcff')

        # plt.title('ROC Scores vs Different Method Values', fontsize=14)
 
    try:
        roc_exp_similarity_plot_data = utils.load_dict('roc_exp_similarity_plot_data_3')  
    
    except (FileNotFoundError, IOError):
        
        nist_plot_dict = utils.load_dict('unique_nist_plot_dict')
        nist_plot_dict_xtb = utils.load_dict('unique_nistxtb_plot_dict')
        nist_plot_dict_pcff = utils.load_dict('unique_nist_plot_dict_pcff')
        # nist_plot_dict_xtbmd = utils.load_dict('unique_nistxtbmd_plot_dict')
        # nist_plot_dict_er = utils.load_dict('unique_nist_plot_dict_er')

        # nist_plot_dict_xtber = utils.load_dict('unique_nistxtb_plot_dict_er')
        # nist_plot_dict_pcffer = utils.load_dict('unique_nist_plot_dict_pcffer')
        
        iden_matrix_quasi = GPUCalculator().pearson_matrix_gpu(nist_plot_dict['exp_list'], nist_plot_dict['pred_list'])
        iden_matrix_dft = GPUCalculator().pearson_matrix_gpu(nist_plot_dict['exp_list'], nist_plot_dict['sim_list'])
        iden_matrix_augxtb = GPUCalculator().pearson_matrix_gpu(nist_plot_dict_xtb['exp_list'], nist_plot_dict_xtb['pred_list'])
        iden_matrix_xtb = GPUCalculator().pearson_matrix_gpu(nist_plot_dict_xtb['exp_list'], nist_plot_dict_xtb['sim_list'])
        iden_matrix_augpcff = GPUCalculator().pearson_matrix_gpu(nist_plot_dict_pcff['exp_list'], nist_plot_dict_pcff['pred_list'])
        iden_matrix_pcff = GPUCalculator().pearson_matrix_gpu(nist_plot_dict_pcff['exp_list'], nist_plot_dict_pcff['sim_list'])
        
        roc_exp_similarity_plot_data = defaultdict(list)
        
        for k in [*range(1, 11, 1), *range(20, 101, 10), *range(200, 1001, 100), *range(2000, 5001, 1000), 4500]:
            print(k)
            
            if len(nist_plot_dict['id_list']) > k:
                top_indices_dft, top_values_dft = extract_top_elements(iden_matrix_dft, k, ex='max')
                top_indices_dft_aug, top_values_dft_aug = extract_top_elements(iden_matrix_quasi, k, ex='max')
            else:
                l = len(nist_plot_dict['id_list'])
                top_indices_dft, top_values_dft = extract_top_elements(iden_matrix_dft, l, ex='max')
                top_indices_dft_aug, top_values_dft_aug = extract_top_elements(iden_matrix_quasi, l, ex='max')
                
            if len(nist_plot_dict_xtb['id_list']) > k:
                top_indices_xtb, top_values_xtb = extract_top_elements(iden_matrix_xtb, k, ex='max')
                top_indices_xtb_aug, top_values_xtb_aug = extract_top_elements(iden_matrix_augxtb, k, ex='max')
            else: 
                m = len(nist_plot_dict_xtb['id_list'])
                top_indices_xtb, top_values_xtb = extract_top_elements(iden_matrix_xtb, m, ex='max')
                top_indices_xtb_aug, top_values_xtb_aug = extract_top_elements(iden_matrix_augxtb, m, ex='max')
                
            if len(nist_plot_dict_pcff['id_list']) > k:
                top_indices_pcff, top_values_pcff = extract_top_elements(iden_matrix_pcff, k, ex='max')
                top_indices_pcff_aug, top_values_pcff_aug = extract_top_elements(iden_matrix_augpcff, k, ex='max')
            else:
                n = len(nist_plot_dict_pcff['id_list'])
                top_indices_pcff, top_values_pcff = extract_top_elements(iden_matrix_pcff, n, ex='max')
                top_indices_pcff_aug, top_values_pcff_aug = extract_top_elements(iden_matrix_augpcff, n, ex='max')
        
            match_similarity_list_quasi = [np.mean(values) for values in top_values_dft_aug]
            match_similarity_list_dft = [np.mean(values) for values in top_values_dft]
            
            match_similarity_list_xtb_aug = [np.mean(values) for values in top_values_xtb_aug]
            match_similarity_list_xtb = [np.mean(values) for values in top_values_xtb]
            
            match_similarity_list_pcff = [np.mean(values) for values in top_values_pcff]
            match_similarity_list_pcff_aug = [np.mean(values) for values in top_values_pcff_aug]
            
            roc_exp_similarity_plot_data['quasi'].append(np.mean(match_similarity_list_quasi))
            roc_exp_similarity_plot_data['augxtb'].append(np.mean(match_similarity_list_xtb_aug))
            roc_exp_similarity_plot_data['augpcff'].append(np.mean(match_similarity_list_pcff_aug))
            
            roc_exp_similarity_plot_data['dft'].append(np.mean(match_similarity_list_dft))
            roc_exp_similarity_plot_data['xtb'].append(np.mean(match_similarity_list_xtb))
            roc_exp_similarity_plot_data['pcff'].append(np.mean(match_similarity_list_pcff))
            
            auc_quasi = plot_roc_by_similarity(iden_matrix_quasi, top_indices_dft_aug, ai_model='Quasi_exp', plot_s='no')
            auc_dft = plot_roc_by_similarity(iden_matrix_dft, top_indices_dft, ai_model='DFT', plot_s='no')
            auc_aug_xtb = plot_roc_by_similarity(iden_matrix_augxtb, top_indices_xtb_aug, ai_model='Aug_xtb', plot_s='no')
            auc_xtb = plot_roc_by_similarity(iden_matrix_xtb, top_indices_xtb, ai_model='xtb', plot_s='no')
            auc_aug_pcff = plot_roc_by_similarity(iden_matrix_augpcff, top_indices_pcff_aug, ai_model='Aug_pcff', plot_s='no')
            auc_pcff = plot_roc_by_similarity(iden_matrix_pcff, top_indices_pcff, ai_model='pcff', plot_s='no')
            
            roc_exp_similarity_plot_data['roc_quasi'].append(auc_quasi)
            roc_exp_similarity_plot_data['roc_dft'].append(auc_dft)
            roc_exp_similarity_plot_data['roc_augxtb'].append(auc_aug_xtb)
            roc_exp_similarity_plot_data['roc_xtb'].append(auc_xtb)
            roc_exp_similarity_plot_data['roc_augpcff'].append(auc_aug_pcff)
            roc_exp_similarity_plot_data['roc_pcff'].append(auc_pcff)
            
        sorted_data = sort_data_by_quasi(roc_exp_similarity_plot_data)    
        utils.save_dict(sorted_data, 'roc_exp_similarity_plot_data_3')  
      
    plot_combined_roc(roc_exp_similarity_plot_data)

#------------------------------------------------------------------------------  

if __name__ == "__main__":
    data_prepare()
    plot_largemol()
    plot_colored_PIDD()
    plot_fgs_iden()
    plot_heatmap() 
    plot_pcc_and_roc_models()
    plotter = SpectrumPlotter(dpi=600)
    analyzer = SpectrumAnalysis(plotter)
    analyzer.compare_xtb_roc()
    analyzer.analyze_dft_data()
    analyzer.analyze_xtb_data(md_data=False)
    analyzer.analyze_xtb_data(md_data=True)
    plot_roc_method()