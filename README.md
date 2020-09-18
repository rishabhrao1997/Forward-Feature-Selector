# Forward-Feature-Selector

The function takes in a set of x_train, y_train, along with other arguments, and iteratively builds input model on the columns one at a time, and keep the column with the highest value of metric. It fixes that selected column and again iteratively builds on that column along with other columns. 
In the end it returns the index of the columns that are the most useful.
