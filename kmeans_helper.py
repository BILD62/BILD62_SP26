from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def clean_PCA_data(data):

    # For ease, store the data and labels in different variables
    brca_data = data.data
    brca_target = data.target

    # Some pre-processing of data
    # Subsample dataset and apply Principal Components Analysis (PCA) [details unimportant for now]
    brca_to_retain = [0, 1, 10, 11]
    brca_data_filtered = brca_data[:,brca_to_retain]

    brca_pca = PCA(n_components=2)

    brca_pca_props = brca_pca.fit(brca_data_filtered)

    print(f'{brca_pca_props.explained_variance_ratio_.sum()*100:.2f}% Variance Explained')

    brca_pca_data = brca_pca.fit_transform(brca_data_filtered)

    return brca_pca_data

def kmeans_plot(data,brca_pca_data,true_labels_str,kmeans_labels_str):

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colour palette
    color_palette = {data.target_names[0]: 'red', data.target_names[1]: 'blue'}
    
    # Plot 1: Actual Target Labels
    sns.scatterplot(x=brca_pca_data[:,0], y=brca_pca_data[:,1], hue=true_labels_str, palette=color_palette, ax=axes[0])
    axes[0].set_title('PCA of Breast Cancer Data (True Labels)')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(title='Diagnosis')
    
    # Plot 2: K-Means Cluster Labels (re-aligned)
    sns.scatterplot(x=brca_pca_data[:,0], y=brca_pca_data[:,1], hue=kmeans_labels_str, palette=color_palette, ax=axes[1])
    axes[1].set_title('PCA of Breast Cancer Data (K-Means Clusters Aligned)')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].legend(title='Cluster Diagnosis')
    
    plt.tight_layout()
    plt.show()