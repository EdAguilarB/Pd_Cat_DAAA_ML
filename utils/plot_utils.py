import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import os
from seaborn import violinplot, stripplot, jointplot, barplot
import shap
from matplotlib import cm
from icecream import ic

def create_st_parity_plot(real, predicted, figure_name, save_path=None):
    """
    Create a parity plot and display R2, MAE, and RMSE metrics.

    Args:
        real (numpy.ndarray): An array of real (actual) values.
        predicted (numpy.ndarray): An array of predicted values.
        save_path (str, optional): The path where the plot should be saved. If None, the plot is not saved.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object.
        matplotlib.axes._axes.Axes: The Matplotlib axes object.
    """
    # Calculate R2, MAE, and RMSE
    r2 = r2_score(real, predicted)
    mae = mean_absolute_error(real, predicted)
    rmse = np.sqrt(mean_squared_error(real, predicted))
    
    # Create the parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(real, predicted, alpha=0.7)
    plt.plot([min(real), max(real)], [min(real), max(real)], color='red', linestyle='--')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    
    # Display R2, MAE, and RMSE as text on the plot
    textstr = f'$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
    plt.gcf().text(0.15, 0.75, textstr, fontsize=12)
    
    # Save the plot if save_path is provided
    if save_path:
        # Ensure the directory exists
        save_path = os.path.join(save_path, figure_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    plt.close()

    

def create_it_parity_plot(real, predicted, index, figure_name, save_path=None):
    r2 = round(r2_score(real, predicted), 3)
    mae = round(mean_absolute_error(real, predicted), 3)
    rmse = round(np.sqrt(mean_squared_error(real, predicted)), 3)


    df = pd.DataFrame({'Real':real,
                       'Predicted': predicted,
                       'Idx': index})

    # Create a scatter plot
    fig = px.scatter(df, x='Real', y='Predicted', text = 'Idx', labels={'x': 'Real Values', 'y': 'Predicted Values'}, hover_data=['Idx', 'Real', 'Predicted'])
    fig.add_trace(go.Scatter(x=real, y=real, mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')))

    # Customize the layout
    fig.update_layout(
        title=f'Parity Plot',
        showlegend=True,
        legend=dict(x=0, y=1),
        xaxis=dict(showgrid=True, showline=True, zeroline=True, linewidth=1, linecolor='black'),
        yaxis=dict(showgrid=True, showline=True, zeroline=True, linewidth=1, linecolor='black'),
        plot_bgcolor='white',  # Set background color to white
    )

    # Display R2, MAE, and RMSE as annotations on the plot
    text_annotation = f'R2 = {r2:.3f}<br>MAE = {mae:.3f}<br>RMSE = {rmse:.3f}'
    fig.add_annotation(
        text=text_annotation,
        xref="paper", yref="paper",
        x=0.15, y=0.75,
        showarrow=False,
        font=dict(size=12),
    )

    # Save the plot as an HTML file if save_path is provided
    if save_path:
        # Ensure the directory exists
        save_path = os.path.join(save_path, figure_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)


def create_training_plot(df, save_path):

    df = pd.read_csv(df)

    epochs = df.iloc[:,0]
    train_loss = df.iloc[:,1]
    val_loss = df.iloc[:,2]
    test_loss = df.iloc[:,3]

    min_val_loss_epoch = epochs[val_loss.idxmin()]

    # Create a Matplotlib figure and axis
    plt.figure(figsize=(10, 6), dpi=300)  # Adjust the figure size as needed
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', linestyle='-')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o', linestyle='-')
    plt.plot(epochs, test_loss, label='Test Loss', marker='o', linestyle='-')

    plt.axvline(x=min_val_loss_epoch, color='gray', linestyle='--', label=f'Min Validation Epoch ({min_val_loss_epoch})')

    # Customize the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(False)

    # Save the plot in high resolution (adjust file format as needed)
    plt.savefig('{}/loss_vs_epochs.png'.format(save_path), bbox_inches='tight')

    plt.close()


def create_bar_plot(means:tuple, stds:tuple, min:float, max:float, metric:str, save_path:str, method1:str, method2:str):

    bar_width = 0.35

    mean_gnn, mean_tml = means
    std_gnn, std_tml = stds

    folds = list(range(1, 11))
    index = np.arange(10)

    plt.bar(index, mean_gnn, bar_width, label=f'{method1} Approach', yerr=std_gnn, capsize=5)
    plt.bar(index+bar_width, mean_tml, bar_width, label=f'{method2} Approach', yerr=std_tml, capsize=5)

    plt.ylim(min*.99, max *1.01)
    plt.xlabel('Fold Used as Test Set', fontsize = 16)

    label = 'Mean $R^2$ Value' if metric == 'R2' else f'Mean {metric} Value'
    plt.ylabel(label, fontsize = 16)

    plt.xticks(index + bar_width / 2, list(folds))

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)

    plt.savefig(os.path.join(save_path, f'{metric}_GNN_vs_TML'), dpi=300, bbox_inches='tight')

    print('Plot {}_GNN_vs_TML has been saved in the directory {}'.format(metric,save_path))

    plt.clf()


def create_violin_plot(data, save_path:str):

    violinplot(data = data, x='Test_Fold', y='Error', hue='Method', split=True, gap=.1, inner="quart", fill=False)

    plt.xlabel('Fold Used as Test Set', fontsize=18)
    plt.ylabel('$\%top_{real}-\%top_{predicted}$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax= plt.gca()
    ax.get_legend().remove()

    plt.savefig(os.path.join(save_path, f'Error_distribution_GNN_vs_TML_violin_plot'), dpi=300, bbox_inches='tight')
    plt.close()


def create_strip_plot(data, save_path:str):

    stripplot(data = data, x='Test_Fold', y='Error', hue='Method', size=3,  dodge=True, jitter=True, marker='D', alpha=.3)

    plt.xlabel('Fold Used as Test Set', fontsize=18)
    plt.ylabel('$\%top_{real}-\%top_{predicted}$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax= plt.gca()
    ax.get_legend().remove()

    plt.savefig(os.path.join(save_path, f'Error_distribution_GNN_vs_TML_strip_plot'), dpi=300, bbox_inches='tight')
    plt.close()

def create_parity_plot(data: pd.DataFrame, save_path:str, method1:str, method2:str):

    results_gnn = data[data['Method'] == method1]

    g = jointplot(x="real_%top", y="predicted_%top", data=results_gnn,
                  kind="reg", truncate=False,
                  xlim=(-1.5, 101), ylim=(-1.5, 101),
                  color="#1f77b4", height=7,
                  scatter_kws={"s": 5, "alpha": 0.3})
    plt.axvline(x=50, color='black', linestyle='--', linewidth=.5)

    # add horizontal line at y=50
    plt.axhline(y=50, color='black', linestyle='--', linewidth=.5)

    plt.text(x=25, y=100, s=f"False Positive", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')
    plt.text(x=75, y=100, s=f"True Positive", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')

    plt.text(x=25, y=45, s=f"True Negative", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')
    plt.text(x=75, y=45, s=f"False Negative", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')

    g.ax_joint.xaxis.label.set_size(20)
    g.ax_joint.yaxis.label.set_size(20)

    g.ax_joint.set_xlabel('Real %top / %')
    g.ax_joint.set_ylabel('Predicted %top / %')

    g.ax_joint.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(os.path.join(save_path, f'parity_plot_{method1}'), dpi=300, bbox_inches='tight')
    plt.close()
    

    results_tml = data[data['Method'] == method2]

    g = jointplot(x="real_%top", y="predicted_%top", data=results_tml,
                  kind="reg", truncate=False,
                  xlim=(-1.5, 101), ylim=(-1.5, 101),
                  color="#ff7f0e", height=7,
                  scatter_kws={"s": 5, "alpha": 0.3})
    plt.axvline(x=50, color='black', linestyle='--', linewidth=.5)

    # add horizontal line at y=50
    plt.axhline(y=50, color='black', linestyle='--', linewidth=.5)

    plt.text(x=25, y=100, s=f"False Positive", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')
    plt.text(x=75, y=100, s=f"True Positive", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')

    plt.text(x=25, y=45, s=f"True Negative", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')
    plt.text(x=75, y=45, s=f"False Negative", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')

    g.ax_joint.xaxis.label.set_size(20)
    g.ax_joint.yaxis.label.set_size(20)

    g.ax_joint.set_xlabel('Real %top / %')
    g.ax_joint.set_ylabel('Predicted %top / %')

    g.ax_joint.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(os.path.join(save_path, f'parity_plot_{method2}'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap(shap_values, X, feat_names, save_path:str):

    plt.figure()
    shap.summary_plot(shap_values, X, max_display = 10, color_bar_label = 'Descriptor value', show = False, plot_size= (14,5.5), feature_names=feat_names)
    plt.grid()
    plt.gcf().axes[-1].set_aspect('auto')
    plt.tight_layout()

    plt.gcf().axes[-1].set_box_aspect(50)
    #Changing plot colours
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(cm.get_cmap('coolwarm'))
    
    plt.savefig(os.path.join(save_path, 'shap_summary_plot'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_importances(df, save_path: str):
    plt.figure(figsize=(10, 6))

    ax = barplot(df, x="score", y="labels", estimator="sum", errorbar=None)
    ax.bar_label(ax.containers[0], fontsize=10)
    # ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center', horizontalalignment='right')

    plt.xlabel('Feature Importance Score', fontsize=16)
    plt.ylabel('Feature', fontsize=16)

    # Save the figure before displaying it
    plt.savefig(os.path.join(save_path, 'node_feature_importance_plot'), dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

    print('Node feature importance plot has been saved in the directory {}'.format(save_path))
    plt.close()

    

