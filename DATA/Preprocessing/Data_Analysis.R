#Data-Driven Clustering of Chronic Low Back Pain Patient Trajectories
#Enhancing Personalized Healthcare and Treatment Planning with SOM-VAE

#Overview of Patient Health Data (refer to Appendix A)


#Load necessary libraries
library(foreign)
library(tidyverse)
library(ggplot2)
library(reshape2)
library(reshape2)
library(plotly)
library(dplyr)

# Prepare working directory set up & LBP dataset
setwd("~/Documents/Ausblidung/LMU/Master/Master Bwl/4.Semester/AI for Goods")
lbp_data <- read.csv("casey_master.csv", sep = "\t", header= TRUE)

# Replace all empty strings with NA
lbp_data[lbp_data == ""] <- NA

# Number of missing values for each variable
missing_values <- sapply(lbp_data, function(x) sum(is.na(x)))
# Total values for each variable (including NAs)
total_values <- sapply(lbp_data, function(x) length(x))

# Combine variable names with missing values count and total values count
summary_stats_df <- data.frame(Variable = names(missing_values), 
                               MissingValues = missing_values,
                               TotalValues = total_values)

# Ensure summary_stats_df is a proper data.frame
summary_stats_df <- as.data.frame(summary_stats_df)
view(summary_stats_df)

# Export the data frame to a CSV file
write.csv(summary_stats_df, file = "summary_stats.csv", row.names = FALSE)
# Notify where the file is saved
cat("The file 'summary_stats.csv' has been saved in the directory:", getwd())

#######
# Identify the top 5 variables with the most missing values
top_5_missing <- summary_stats_df %>% 
  arrange(desc(MissingValues)) %>% 
  head(5)

# Ensure top_5_missing is a proper data.frame
top_5_missing <- as.data.frame(top_5_missing)
# Display the top 5 variables with the most missing values
print(top_5_missing)

# Export the data frame to a CSV file
write.csv(top_5_missing, file = "top_5_missing.csv", row.names = FALSE)

# Visualize the top 5 variables with the most missing values
ggplot(top_5_missing, aes(x = reorder(Variable, -MissingValues), y = MissingValues)) +
  geom_bar(stat = "identity", fill = "#2c287f") +
  geom_text(aes(label = MissingValues), vjust = -0.5, color = "black", size = 5) +
  xlab("Variable") +
  ylab("Number of Missing Values") +
  ggtitle("Top 5 Variables with Most Missing Values") +
  theme_bw(base_size = 15) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  scale_y_continuous(limits = c(0, max(top_5_missing$MissingValues) * 1.1)) +
  theme(
    panel.grid.major = element_line(color = "gray85"),
    panel.grid.minor = element_blank()
  )

#######
# Select 8 specified variables from our LBP dataset
summary_stats_8v <- lbp_data %>% select(gen12m, rmprop12m, vasl0, bsex0, bhoej0, bmi, age, htil0)

# Convert bsex0 to numeric: female = 1, male = 0
summary_stats_8v <- summary_stats_8v %>%
  mutate(bsex0 = case_when(
    bsex0 == "female" ~ 1,
    bsex0 == "male" ~ 0,
    TRUE ~ NA_real_  # Handle any other values or NA
  ))

# Convert gen12m to numeric: based on the provided mapping
summary_stats_8v <- summary_stats_8v %>%
  mutate(gen12m = case_when(
    gen12m == "Completely recovered" ~ 7,
    gen12m == "Much improved" ~ 6,
    gen12m == "Slightly improved" ~ 5,
    gen12m == "Not changed" ~ 4,
    gen12m == "Slightly worsened" ~ 3,
    gen12m == "Much worsened" ~ 2,
    gen12m == "Worse than ever" ~ 1,
    TRUE ~ NA_real_ 
  ))

# Ensure all selected variables are numeric
summary_stats_8v <- summary_stats_8v %>% mutate(across(everything(), as.numeric))
str(summary_stats_8v)

# Calculate summary statistics for each variable
summary_stats_df_8v <- summary_stats_8v %>%
  summarise(across(everything(), list(mean = ~mean(., na.rm = TRUE), 
                                      sd = ~sd(., na.rm = TRUE), 
                                      min = ~min(., na.rm = TRUE), 
                                      max = ~max(., na.rm = TRUE), 
                                      median = ~median(., na.rm = TRUE), 
                                      na_count = ~sum(is.na(.)))))

# Reshape the summary statistics data frame
summary_stats_long_8v <- summary_stats_df_8v %>%
  pivot_longer(everything(), names_to = c("Variable", ".value"), names_sep = "_")
view(summary_stats_long_8v)

# Ensure summary_stats_long_8v is a proper data.frame
summary_stats_long_8v <- as.data.frame(summary_stats_long_8v)

# Export the data frame to a CSV file
write.csv(summary_stats_long_8v, file = "summary_stats_long_8v.csv", row.names = FALSE)

#######
# Convert bsex0 to numeric: female = 1, male = 0
lbp_data <- lbp_data %>%
  mutate(bsex0 = case_when(
    bsex0 == "female" ~ 1,
    bsex0 == "male" ~ 0,
    TRUE ~ NA_real_  # Handle any other values or NA
  ))

# Convert gen12m to numeric
lbp_data <- lbp_data %>%
  mutate(gen12m = case_when(
    gen12m == "Completely recovered" ~ 7,
    gen12m == "Much improved" ~ 6,
    gen12m == "Slightly improved" ~ 5,
    gen12m == "Not changed" ~ 4,
    gen12m == "Slightly worsened" ~ 3,
    gen12m == "Much worsened" ~ 2,
    gen12m == "Worse than ever" ~ 1,
    TRUE ~ NA_real_  # Handle any other values or NA
  ))

#######
# Missing values heatmap
library(Amelia)
missmap(lbp_data, vars=15, main = "Missing Values Heatmap")
missmap(lbp_data, main = "Missing Values Heatmap", col = c("lightgray", "darkblue"), legend = TRUE)

#######
# Remove NA values from gen12m
lbp_data <- lbp_data %>% filter(!is.na(gen12m))

# Convert gen12m to a factor with labels
lbp_data$gen12m <- factor(lbp_data$gen12m, levels = 1:7, 
                          labels = c("Worse than ever", "Much worsened", "Slightly worsened", 
                                     "Not changed", "Slightly improved", "Much improved", 
                                     "Completely recovered"))

#######
# Create the bar plot
ggplot(lbp_data, aes(x = gen12m)) +
  geom_bar(fill = "#2c287f", color = "black") +
  theme_bw() +
  xlab("General Health Improvement of LBP Patients") +
  ylab("Count") +
  ggtitle("Distribution of General Health Improvement") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  scale_y_continuous(limits = c(0, max(table(lbp_data$gen12m)) * 1.1)) +
  theme(
    panel.grid.major = element_line(color = "gray85"),
    panel.grid.minor = element_blank()
  ) +
  guides(fill = guide_legend(title = "Labels", 
                             title.position = "top",
                             title.hjust = 0.5,
                             override.aes = list(fill = "#2c287f")))

#######
# Scatter plot with jitter and regression line
ggplot(lbp_data, aes(x = age, y = bmi, color = gen12m)) +
  geom_jitter(width = 0.3, height = 0.3, size = 2, alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  scale_color_manual(values = c("darkred", "red", "orange", "yellow", "lightgreen", "green", "darkgreen")) +
  theme_bw() +
  xlab("Age") +
  ylab("BMI") +
  ggtitle("Exploring the Influence of Age on BMI Across Different Health Improvement Levels") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    legend.title = element_text(face = "bold"),
    legend.position = "bottom"
  ) +
  labs(color = "General Health Improvement")

ggplot(lbp_data, aes(x = htil0, y = bmi, color = gen12m)) +
  geom_jitter(width = 0.3, height = 0.3, size = 2, alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  scale_color_manual(values = c("darkred", "red", "orange", "yellow", "lightgreen", "green", "darkgreen")) +
  theme_bw() +
  xlab("Self-rated General Health") +
  ylab("BMI") +
  ggtitle("Exploring the Influence of Self-rated General Health on BMI Across Different Health Improvement Levels") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    legend.title = element_text(face = "bold"),
    legend.position = "bottom"
  ) +
  labs(color = "General Health Improvement")

#######
####BMI
# Box plot for BMI across General Health Improvement categories
ggplot(lbp_data, aes(x = gen12m, y = bmi, fill = gen12m)) +
  geom_boxplot() +
  scale_fill_manual(values = c("darkred", "red", "orange", "yellow", "lightgreen", "green", "darkgreen")) +
  theme_bw() +
  xlab("General Health Improvement") +
  ylab("BMI") +
  ggtitle("Comparison of BMI Among Different Health Improvement Levels") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"  # Remove legend as colors are self-explanatory
  )

# Box plot for BMI across General Health Improvement categories with jitter and min-max lines
ggplot(lbp_data, aes(x = gen12m, y = bmi, fill = gen12m)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.2, size = 1, alpha = 0.6) +
  stat_summary(fun = min, geom = "point", shape = 95, size = 4, color = "black", position = position_dodge(width = 0.75)) +
  stat_summary(fun = max, geom = "point", shape = 95, size = 4, color = "black", position = position_dodge(width = 0.75)) +
  scale_fill_manual(values = c("darkred", "red", "orange", "yellow", "lightgreen", "green", "darkgreen")) +
  theme_bw() +
  xlab("General Health Improvement") +
  ylab("BMI") +
  ggtitle("Comparison of BMI Among Different Health Improvement Levels") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"  # Remove legend as colors are self-explanatory
  )

#######
#### HEIGHT
# Box plot for Height (bhoej0) across General Health Improvement categories
ggplot(lbp_data, aes(x = gen12m, y = bhoej0, fill = gen12m)) +
  geom_boxplot() +
  scale_fill_manual(values = c("darkred", "red", "orange", "yellow", "lightgreen", "green", "darkgreen")) +
  theme_bw() +
  xlab("General Health Improvement") +
  ylab("Height") +
  ggtitle("Box Plot of Height by General Health Improvement") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"  # Remove legend as colors are self-explanatory
  )

# Box plot for Height (bhoej0) across General Health Improvement categories with jitter and min-max lines
ggplot(lbp_data, aes(x = gen12m, y = bhoej0, fill = gen12m)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.2, size = 1, alpha = 0.6) +
  stat_summary(fun = min, geom = "point", shape = 95, size = 4, color = "black", position = position_dodge(width = 0.75)) +
  stat_summary(fun = max, geom = "point", shape = 95, size = 4, color = "black", position = position_dodge(width = 0.75)) +
  scale_fill_manual(values = c("darkred", "red", "orange", "yellow", "lightgreen", "green", "darkgreen")) +
  theme_bw() +
  xlab("General Health Improvement") +
  ylab("Height") +
  ggtitle("Comparison of Height Among Different Health Improvement Levels") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"  # Remove legend as colors are self-explanatory
  )

#######
#### AGE
# Box plot for Age across General Health Improvement categories
ggplot(lbp_data, aes(x = gen12m, y = age, fill = gen12m)) +
  geom_boxplot() +
  scale_fill_manual(values = c("darkred", "red", "orange", "yellow", "lightgreen", "green", "darkgreen")) +
  theme_bw() +
  xlab("General Health Improvement") +
  ylab("Age") +
  ggtitle("Comparison of Age Among Different Health Improvement Levels") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"  # Remove legend as colors are self-explanatory
  )

# Box plot for Age across General Health Improvement categories with jitter and min-max lines
ggplot(lbp_data, aes(x = gen12m, y = age, fill = gen12m)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.2, size = 1, alpha = 0.6) +
  stat_summary(fun = min, geom = "point", shape = 95, size = 4, color = "black", position = position_dodge(width = 0.75)) +
  stat_summary(fun = max, geom = "point", shape = 95, size = 4, color = "black", position = position_dodge(width = 0.75)) +
  scale_fill_manual(values = c("darkred", "red", "orange", "yellow", "lightgreen", "green", "darkgreen")) +
  theme_bw() +
  xlab("General Health Improvement") +
  ylab("Age") +
  ggtitle("Comparison of Age Among Different Health Improvement Levels") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"  # Remove legend as colors are self-explanatory
  )

#######
#####AGE ANALYSIS
# Remove NA values from age
lbp_data_age <- lbp_data %>% filter(!is.na(age))

# Define age groups
lbp_data_age$age_group <- cut(lbp_data_age$age, 
                               breaks = c(17, 25, 32, 39, 46, 53, 66), 
                               labels = c("18-25 years", "26-32 years", "33-39 years", "40-46 years", "47-53 years", "54-65 years"), 
                               right = FALSE)

# Calculate counts for each age group
age_group_counts <- as.data.frame(table(lbp_data_age$age_group))

# Histogram for age groups with counts
ggplot(lbp_data_age, aes(x = age_group)) +
  geom_histogram(stat = "count", fill = "#2c287f", color = "black") +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  labs(title = "Histogram of Age Groups", x = "Age Group", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(lbp_data_age, aes(x = age_group)) +
  geom_histogram(stat = "count", fill = "#2c287f", color = "black") +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  labs(title = "Age Group Distribution of LBP Patients", x = "Age Group", y = "Count") +
  theme_bw() + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1))

#######
#####GENDER ANALYSIS
# Convert bsex0 to a factor with labels
lbp_data$bsex0 <- factor(lbp_data$bsex0, levels = c(0, 1), labels = c("Male", "Female"))

# Create the histogram plot for gender with a legend
ggplot(lbp_data, aes(x = bsex0)) +
  geom_bar(fill = "#2c287f", color = "black") +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  labs(title = "Gender Distribution of LBP Patients", x = "Gender", y = "Count") +
  theme_bw() + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )