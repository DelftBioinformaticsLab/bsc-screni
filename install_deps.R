
install.packages("BiocManager", repos = "https://cloud.r-project.org")
BiocManager::install("rhdf5")  # SeuratDisk needs this

install.packages("remotes", repos = "https://cloud.r-project.org")
remotes::install_github("mojaveazure/seurat-disk")
