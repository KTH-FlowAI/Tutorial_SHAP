-----------------------------------------------------------------------------------------------------------------
Tutorial for SHAP application in image classification
-----------------------------------------------------------------------------------------------------------------

This tutorial is a modification of the presented tutorial in https://shap.readthedocs.io/en/latest/index.html
It is based on the work of Scott Lundberg presented in the repository: https://github.com/shap/shap.git 

More details in the following references by our group:

- Cremades, A., Hoyas, S., Deshpande, R., Quintero, P., Lellep, M., Lee, W. J., ... & Vinuesa, R. (2024). Identifying regions of importance in wall-bounded turbulence through explainable deep learning. Nature Communications, 15(1), 3864.https://www.nature.com/articles/s41467-024-47954-6

- Cremades, A., Hoyas, S., & Vinuesa, R. (2025). Additive-feature-attribution methods: a review on explainable artificial intelligence for fluid dynamics and heat transfer. International Journal of Heat and Fluid Flow, 112, 109662.https://www.sciencedirect.com/science/article/pii/S0142727X24003874

- Cremades, A., Hoyas, S., & Vinuesa, R. (2024). Classically studied coherent structures only paint a partial picture of wall-bounded turbulence. arXiv preprint arXiv:2410.23189.https://arxiv.org/abs/2410.23189

-----------------------------------------------------------------------------------------------------------------

This repository modifies the original one to adapt the problem of calculating MNIST predictions through 
gradientExplainer to do it also for kernelExplainer segmenting the domain and deepExplainer.

-----------------------------------------------------------------------------------------------------------------
