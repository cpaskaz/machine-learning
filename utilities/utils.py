import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec
import math
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, recall_score, roc_curve, auc, accuracy_score

# To build models for prediction
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

###################################################################
# General helper functions                                        #
###################################################################

# Function to delete a Dataframe from memory
def delete_dataframe_if_exists(df_name):
    try:
        # Try to access the variable. If it exists, proceed to delete.
        globals()[df_name]  # Or use locals() if within a local scope
        del globals()[df_name]  # Or use locals() if within a local scope
        print(f"{df_name} DataFrame deleted.")
    except KeyError:
        # If the DataFrame does not exist, print a message.
        print(f"{df_name} DataFrame does not exist.")

# Function to check if a DataFrame exists, and if not, create and append data
def check_create_append_df(existing_df_name, data):
    new_df = pd.DataFrame(data)
    # Check if DataFrame exists in the local variables
    if existing_df_name in globals():
        # DataFrame exists, append the data
        globals()[existing_df_name] = pd.concat([globals()[existing_df_name], new_df], ignore_index=True)
    else:
        # DataFrame does not exist, create it with the data
        globals()[existing_df_name] = pd.DataFrame(data)

    return globals()[existing_df_name]
    
###################################################################
# Plotting helper functions                                       #
###################################################################

#function to add percentages to countplot bars with hue enabled
def with_hue(ax, feature, plot_categories, hue_categories):
    a = [p.get_height() for p in ax.patches]
    patch = [p for p in ax.patches]
    for i in range(plot_categories):
        total = feature.value_counts().values[i]
        for j in range(hue_categories):
          percentage = '{:.1f}%'.format(100 * a[(j * plot_categories + i)]/total)
          #print(feature, total, percentage )
          x = patch[(j * plot_categories + i)].get_x() + patch[(j * plot_categories + i)].get_width() / 2 - 0.1
          y = patch[(j * plot_categories + i)].get_y() + patch[(j * plot_categories + i)].get_height() + 2
          ax.annotate(percentage, (x, y), size = 7, ha="left", va="center", xytext=(0, 5), textcoords="offset points")

#function to add percentages to countplot bars with no hue enabled
def without_hue(ax, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.1
        y = p.get_y() + p.get_height() + 2
        ax.annotate(percentage, (x, y), size = 7, ha="left", va="center", xytext=(0, 5), textcoords="offset points")

#function to calculate the shape of a grid for plotting several charts at once        
def calculate_subplot_grid_shape(df, max_columns):
    #---------------------------------------------------------------------------------------------------------------------------------
    # Purpose: Calculates the subplot grid shape (rows, columns) based on the number of rows in a list.                               
    # Parameters: 
    #       - num_col: list of numerical columns to plot 
    #       - data: datframe where the data is
    #       - max_columns: maximum columns you want in the matrix  
    # Returns: A tuple (n_rows, n_cols) indicating the shape of the grid.
    #---------------------------------------------------------------------------------------------------------------------------------
    
    n_plots = len(df)
    
    # Calculate the number of columns by finding the square root and rounding up
    #n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_cols = min(int(np.ceil(np.sqrt(n_plots))), max_columns)
    
    # Calculate the number of rows needed to accommodate all plots
    n_rows = int(np.ceil(n_plots / n_cols))
    
    return n_rows, n_cols

#function to plot a boxplot on top of a histogram and share the x-axis    
def histogram_boxplot(feature, title, figsize=(10, 6), bins="auto"):
    ''' Boxplot and histogram combined feature: 1-d feature array
        title: the title given to the plot figsize: size of fig (default (15, 10))
        bins: number of bins (default "auto") '''

    f, (ax_box, ax_hist) = plt.subplots(
        nrows=2,                                    # Number of rows of the subplot grid
        sharex=True,                                # The X-axis will be shared among all the subplots
        gridspec_kw={"height_ratios": (.25, .75)},  # grid layout to place subplots within the figure
        figsize=figsize)                            # size of the plot

    ########################
    # Creating the subplots
    ########################

    # Boxplot
    sns.boxplot(x=feature, ax=ax_box, showmeans=True)                 # create boxplot, show mean value of the column will be indicated using some symbol
    ax_box.set_title('Distribution Charts for: ' + title)             # add the title

    # Histogram
    sns.histplot(x=feature, kde=True, ax=ax_hist, bins=bins)          # create histogram, show kde
    ax_hist.axvline(np.mean(feature), color='g', linestyle='--')      # Add mean to the histogram
    ax_hist.axvline(np.median(feature), color='black', linestyle='-') # Add median to the histogram

    plt.show()  # display the plot

#function to plot a boxplot on top of a histogram and share the x-axis, in a grid format (several in one plot)
def histogram_boxplot_grid(num_col, data, max_columns):
    #---------------------------------------------------------------------------------------------------------------------------------
    # Purpose: This plot will create a matrix of boxplots and histograms that sit on top of one another.                              
    # Parameters: 
    #       - num_col: list of numerical columns to plot
    #       - data: datframe where the data is
    #       - max_columns: maximum columns in matrix  
    #---------------------------------------------------------------------------------------------------------------------------------
    
    gridrow = 0
    n_rows, n_cols = calculate_subplot_grid_shape(df=num_col, max_columns=3)

    # Set up the figure and GridSpec
    fig = plt.figure(figsize=(6*n_cols, 4*n_rows))  # Adjust figure size dynamically based on the number of columns

    #set the height ratios of the three plots
    hratio = []
    for i in range(n_rows):
        hratio.append(1)
        hratio.append(6)
        hratio.append(2)

    # set the grid spec
    gs = gridspec.GridSpec(n_rows*3, n_cols, height_ratios=hratio, hspace=0.05, wspace=0.2)

    # loop through the list of variables you want to plot
    for i, g in enumerate(num_col):
        mod = i % n_cols

        # Empty plot at the bottom to give spacing after every second row
        ax_blank = plt.subplot(gs[gridrow+2, mod])
        ax_blank.spines['top'].set_visible(False)
        ax_blank.spines['right'].set_visible(False)
        ax_blank.spines['left'].set_visible(False)
        ax_blank.spines['bottom'].set_visible(False)
        ax_blank.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # Top row for boxplot
        ax_box = plt.subplot(gs[gridrow, mod])
        sns.boxplot(x=data[g], ax=ax_box, showmeans=True, palette='coolwarm')
        ax_box.set(yticklabels=[])
        ax_box.xaxis.set_visible(False)  # Hide the x-axis
        ax_box.yaxis.set_visible(False)  # Hide the y-axis

        # Bottom row for histograms, sharing the x-axis with the boxplot
        ax_hist = plt.subplot(gs[gridrow+1, mod], sharex=ax_box)
        sns.histplot(x=data[g], kde=True, ax=ax_hist, bins=50, palette='coolwarm', )         
        ax_hist.axvline(np.mean(data[g]), color='g', linestyle='--')      
        ax_hist.axvline(np.median(data[g]), color='black', linestyle='-') 
        ax_hist.set_xlabel(f'{g}')
    
        # Add to the counter when we are ready to start a new gridrow
        if mod == (n_cols-1):
            gridrow = gridrow + 3 # there are three plots, so the counter needs to be incremented by three

    plt.suptitle('Boxplot/Histogram Matrix') # set the title
    plt.show()   

#function to plot a countplot ordered with percenatges on top of the bar    
def df_countplot(data, feature, orderval, topn=10, figsize=(10, 6)):
    # create a countplot and only show the top 5
    ax = sns.countplot(x=feature, data=data, palette='Paired', order=df[orderval].value_counts().iloc[:topn].index, figsize=figsize)
    total = len(df[feature])                                        # Length of the column

    for p in ax.patches:                                            # loop through the category / patches to perform some custom graphing
      percentage = '{:.1f}%'.format(100 * p.get_height() / total) # Percentage of each class
      x = p.get_x() + p.get_width() / 2 - .1                      # Width of the plot
      y = p.get_y() + p.get_height() + 2                          # Height of the plot
      ax.annotate(percentage, (x, y), size = 6)                   # Annotate the percentage

    plt.title('Countplot for: ' + feature + ' - Top ' + str(topn))       # set the title
    plt.tick_params(axis='x', rotation=90, labelsize = 10)          # rotate the tick labels
    plt.show()

#function to plot a countplot grid
def countplot_grid(data, cols, hue=False, var=None, showpct=True, max_columns=4):
    # Calculate grid shape
    n_rows, n_cols = calculate_subplot_grid_shape(df=cols, max_columns = max_columns)
  
    # Create the figure and two subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))

    # Flatten the axes array for easy indexing
    axes_flat = axes.flatten()


    # Loop through the DataFrame rows and create a boxplot in each subplot
    for ind, x in enumerate(cols):
      if hue and var: 
        ax = sns.countplot(ax=axes_flat[ind], x=data[x], hue=data[var], data=pd.melt(data), palette='coolwarm')
        n_cat = data[var].nunique()
        n_hue = data[x].nunique()
        ax.legend(loc='upper left', bbox_to_anchor=(0.35, 1.2), shadow=False, ncol=4, frameon=False, fontsize=7)
        ax.set_title(f'Chart for: {x} by {var}', pad=30, fontsize=10, weight='bold')
      if not hue: 
        ax = sns.countplot(ax=axes_flat[ind], x=data[x], data=pd.melt(data), palette='coolwarm') 
        ax.set_title(f'Chart for: {x}', pad=30, fontsize=10, weight='bold') 
      
      ax.tick_params(axis='x', rotation=90, labelsize=7)  # rotate the axis
      ax.tick_params(axis='y', labelsize=7)  # rotate the axis
      ax.xaxis.label.set_size(8)
      ax.yaxis.label.set_size(8)
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      #ax.set_title(f'Chart for: {x} by {var}', pad=30, fontsize=10, weight='bold')
      #ax.title.set_size(10) 

      if showpct and hue: with_hue(ax=ax, feature=data[var], plot_categories=n_cat, hue_categories=n_hue)
      if showpct and not hue: without_hue(ax=ax, feature=data[x]) 

    # If there are any remaining empty subplots, turn them off
    for i in range(n_rows * n_cols):
        if i > (len(cols) - 1):  
            axes_flat[i].axis('off')
  
    # Show the plot
    plt.suptitle('Countplot Matrix', fontsize=14, weight='bold') # set the title
    plt.tight_layout(pad=3.0)
    plt.show()

#function to plot a pair plot
def pplot(data, vars):
    # use a pair plot to look at the distrubution and the correlation between the numeric variables
    sns.pairplot(data=data, vars=vars, corner=True, diag_kind="kde")

    # set the title and show the plot
    plt.suptitle("Pairplot of the Numerical Variables")
    plt.show()

#function to plot a bar plot with percentages on top    
def bar_perc(data, z, hue=False, hue_column=None, orientation='V', figsize=(10,6), showpct=True):
    total = len(data[z]) # Length of the column
    plt.figure(figsize = figsize)

    # Convert the column to a categorical data type
    #data[z] = data[z].astype('category')

    if orientation == 'H':
      if hue: ax = sns.countplot(y=z, data=data, palette='coolwarm', order=data[z].value_counts().index, hue=hue_column)
      if not hue: ax = sns.countplot(y=z, data=data, palette='coolwarm', order=data[z].value_counts().index)
      if showpct:
        for p in ax.patches:                                           # add percentage labels
          percentage = "{:.1f}%".format(100.0 * p.get_width() / total)    # calculated teh percentage
          y = p.get_y() + p.get_height() / 2                            # y coordinate of bar percentage label
          x = p.get_width()                                             # x coordinate of bar percentage label
          ax.annotate(percentage,(x, y), ha="left", va="center", size=8, xytext=(5, 0), textcoords="offset points")  # plot the percentage
          ax.set_title('Count Chart for: ' + z)           # Set the Title of the graph
    else:
      if hue: ax = sns.countplot(x=z, data=data, palette='coolwarm', order=data[z].value_counts().index, hue=hue_column)
      if not hue: ax = sns.countplot(x=z, data=data, palette='coolwarm', order=data[z].value_counts().index)
      if showpct:
        for p in ax.patches:
          percentage = '{:.1f}%'.format(100 * p.get_height() / total) # Percentage of each class
          x = p.get_x() + p.get_width() / 2 - .1                      # Width of the plot
          y = p.get_y() + p.get_height() + 2                          # Height of the plot
          ax.annotate(percentage, (x, y), size = 8, ha="left", va="center", xytext=(5, 5), textcoords="offset points")                   # Annotate the percentage
          ax.set_title('Count Chart for: ' + z)           # Set the Title of the graph
          ax.tick_params(axis='x', rotation=90, labelsize = 8)        # Rotate the axis for read the lables
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    plt.show()

#function to plot a bar plot grid with percentages
def bar_perc_grid(data, var, hue=False, hue_column=None, orientation='V', showpct=True):
    total = len(data[var]) # Length of the column
    plt.figure(figsize = figsize)

    # Convert the column to a categorical data type
    #data[var] = data[var].astype('category')

    if orientation == 'H':
      if hue: ax = sns.countplot(y=var, data=data, palette='Paired', order=data[var].value_counts().index, hue=hue_column)
      if not hue: ax = sns.countplot(y=var, data=data, palette='Paired', order=data[var].value_counts().index)
      if showpct:
        for p in ax.patches:                                            # add percentage labels
          percentage = "{:.1f}%".format(100.0*p.get_width() / total)    # calculated teh percentage
          y = p.get_y() + p.get_height() / 2                            # y coordinate of bar percentage label
          x = p.get_width()                                             # x coordinate of bar percentage label
          ax.annotate(percentage,(x, y), ha="left", va="center", size=8, xytext=(5, 0), textcoords="offset points")  # plot the percentage
          #ax.set_title('Distribution Charts for: ' + title)           # Set the Title of the graph
    else:
      if hue: ax = sns.countplot(x=var, data=data, palette='Paired', order=data[var].value_counts().index, hue=hue_column)
      if not hue: ax = sns.countplot(x=var, data=data, palette='Paired', order=data[var].value_counts().index)
      if showpct:
        for p in ax.patches:
          percentage = '{:.1f}%'.format(100 * p.get_height() / total) # Percentage of each class
          x = p.get_x() + p.get_width() / 2 - .1                      # Width of the plot
          y = p.get_y() + p.get_height() + 2                          # Height of the plot
          ax.annotate(percentage, (x, y), size = 6)                   # Annotate the percentage
          ax.set_title('Distribution Charts for: ' + z)           # Set the Title of the graph
          ax.tick_params(axis='x', rotation=90, labelsize = 6)        # Rotate the axis for read the lables
    plt.show()

#function to plot calculate and plot a correlation matrix    
def corr_matrix(data):
    corr = data.corr() # <- Dont include the status column as it is boolean

    # Plot the correltion heatmap
    plt.figure(figsize = (10, 6))
    sns.heatmap(corr, annot = True, cmap = 'coolwarm', fmt = ".3f",
        xticklabels = corr.columns,
        yticklabels = corr.columns)
    # set the title and show the plot
    plt.title("Heatmap of the Correlation Matrix for the Numerical Values")
    plt.show()

#function to plot a boxplot in grid format (several in one plot)    
def boxplot_grid(pairinput, df):
  # Calculate grid shape
  n_rows, n_cols = calculate_subplot_grid_shape(df=pairinput, max_columns=3)
  #print(n_rows, n_cols)
  
  # Create the figure and two subplots
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
  #fig.tight_layout(pad=3.0)
  
  # Flatten the axes array for easy indexing
  axes_flat = axes.flatten()

  # Loop through the DataFrame rows and create a boxplot in each subplot
  for ind in pairinput.index:
    #print(ind, axes_flat[ind], pairinput['num_column'][ind], pairinput['cat_column'][ind])
    ax = sns.boxplot(ax=axes_flat[ind], data=df, y=pairinput['num_column'][ind], x=pairinput['cat_column'][ind], palette='coolwarm') 
    ax.tick_params(axis='x', rotation=90, labelsize=7)  # rotate the axis
    ax.tick_params(axis='y', labelsize=7)  # rotate the axis
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    
  # If there are any remaining empty subplots, turn them off
  for i in range(n_rows * n_cols):
    if i > (len(pairinput) - 1):  
      axes_flat[i].axis('off')
  
  # Show the plot
  plt.suptitle('Boxplot Matrix') # set the title
  plt.tight_layout()
  plt.show()

# Define a stacked bar plot to compare each feature's relationship to status.
def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 100)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

###################################################################
# Statistics / Distribution helper functions                      #
###################################################################

#function to test a distribution is statitically different based on categories (Anova, Kruskal-Wallace etc..)
def dist_different(df, category, value):
  # Group by category
  grouped = df.groupby(category)[value]

  # Step 1: Test for Normality in each group
  normality_tests = {cat: stats.shapiro(group)[1] for cat, group in grouped}

  #print("Normality Tests:")
  for cat, p_value in normality_tests.items():
    #print(f"Category {cat}: Shapiro-Wilk test p-value = {p_value}")
    a = 1

    # Assuming all groups are normal if their p-values > 0.05
    all_normal = all(p > 0.05 for p in normality_tests.values())

    if len(normality_tests) == 2:
      # For two categories, use t-test or Mann-Whitney depending on normality
      data_groups = [group for _, group in grouped]
      if all_normal:
        test_result = stats.ttest_ind(*data_groups, equal_var=False)
        stype = 'T-test'
        stat = test_result[0]
        p_val = test_result[1]
        #print("\nT-test (assuming unequal variances):", test_result)
      else:
        test_result = stats.mannwhitneyu(*data_groups)
        stype = 'Mann-Whitney U Test'
        stat = test_result[0]
        p_val = test_result[1]
        #print("\nMann-Whitney U Test:", test_result)
    else:
    # For more than two categories, use ANOVA or Kruskal-Wallis depending on normality
      if all_normal:
        # ANOVA assumes equal variances, so we check this first
        f_val, p_val_anova = stats.f_oneway(*[group for _, group in grouped])
        stype = 'Anova'
        stat = f_val
        p_val = p_val_anova      
        #print("\nANOVA Test:", f"F-value = {f_val}, p-value = {p_val_anova}")
      else:
        # Kruskal-Wallis does not assume normality or equal variances
        k_val, p_val_kruskal = stats.kruskal(*[group for _, group in grouped])
        stype = 'Kruskal-Wallis'
        stat = k_val
        p_val = p_val_kruskal
        #print("\nKruskal-Wallis Test:", f"Kruskal statistic = {k_val}, p-value = {p_val_kruskal}")
  return (value, category, len(normality_tests), stype, stat, p_val)

#function to test a distribution and save results to a data frame
def distribution_check(data, num_col, cat_col, alpha=.05):
  # Setup empty dataframe to capture the distribution test results
  normdf = pd.DataFrame(columns=['num_column', 'cat_column', 'num_categories','stat_test', 'stat_val','p_val', 'is_different'])
  
  # Loop through the numeric columns and category columns to test differences in distributions
  for y in num_col: 
    for x in cat_col: 
      result = dist_different(data, x,  y)
      if result[5] <= alpha:
        new_row = {'num_column': result[0], 'cat_column': result[1], 'num_categories': result[2], 'stat_test': result[3], 'stat_val': result[4], 'p_val': result[5], 'is_different': True}
        normdf = pd.concat([normdf, pd.DataFrame([new_row])], ignore_index=True)
      else:
        new_row = {'num_column': result[0], 'cat_column': result[1], 'num_categories': result[2], 'stat_test': result[3], 'stat_val': result[4], 'p_val': result[5], 'is_different': False}
        normdf = pd.concat([normdf, pd.DataFrame([new_row])], ignore_index=True)
  return normdf

# Function to remove outliers from selected columns using the IQR method
def remove_outliers(df, columns, cleantype='IQR'):
    cleaned_df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    if cleantype == 'IQR':
        for column in columns:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identifying outliers and filtering them out
                cleaned_df = cleaned_df[(cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)]
            else:
                print(f"Column '{column}' not found in DataFrame. Skipping.")
    
    return cleaned_df
    
###################################################################
# Machine Learning helper functions                               #
###################################################################

#function to plot a ROC-AUC curve
def metrics_roc(y_ds, y_score, plot=True):
  # Create ROC-AUC parameters
  fpr, tpr, _ = metrics.roc_curve(y_ds, y_score)
  roc_auc = metrics.auc(fpr, tpr)

  if plot:
    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

  return roc_auc
  
# Function to evaluate and create a dataframe to hold the model metrics, plot the confustion matrix and Important features
def evaluate_classification_metrics(model, actual, predicted, x_ds, trained_model):
    # Generating the confusion matrix
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    
    # Calculating the metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    y_score = trained_model.predict_proba(x_ds)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(actual, y_score)
    
    roc_auc = metrics.auc(fpr, tpr)
    #auc = metrics_roc(actual, y_score, False)
    
    dta = {'Model': [model], 'Precision': [precision], 'Recall': [recall],'F1-Score': [f1_score],'Accuracy': [accuracy], 'ROC-AUC': [roc_auc]}
    df = pd.DataFrame(dta)
    
    df_name = 'dfmr'
    ds = check_create_append_df(df_name, dta)
    
    ##### Start the output and plots #####
    print(df.to_string(index=False))
    print('')

    # Creating the plot grid
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2)  # Define a grid of 3 rows, 2 columns

    # Plot the confusion matrix
    cm = confusion_matrix(actual, predicted)
    
    ax1 = fig.add_subplot(gs[0, 0])  # This places it in grid cell (1, 1)
    sns.heatmap(ax=ax1, data=cm, annot=True,  fmt='.2f', xticklabels=['0', '1'], yticklabels=['0', '1'], cmap="coolwarm")
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')
    ax1.set_title('Confusion Matrix')
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

    # Plotting the ROC curve
    ax2 = fig.add_subplot(gs[1, 0])  # This places it in grid cell (2, 1)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)    
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend(loc="lower right")
    #plt.show()

    # Plot feature importances
    importances = trained_model.feature_importances_
    columns = x_ds.columns
    importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)
    
    ax3 = fig.add_subplot(gs[:, 1])  # This makes it span all rows in column 2
    sns.barplot(ax=ax3, x=importance_df.Importance,y=importance_df.index, orient='h', palette='coolwarm')
    ax3.set_title('Feature Importances')
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Importance')
    ax3.tick_params(axis='x', labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.show()
    
    return ds