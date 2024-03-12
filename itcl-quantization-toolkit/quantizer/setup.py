from setuptools import setup
packages = \
['itcl_quantizer',
# 'itcl_quantizer.equalizers',
# 'itcl_quantizer.equalizers.adaround',
# 'itcl_quantizer.equalizers.param_equalizer',
# 'itcl_quantizer.interfaces',
# 'itcl_quantizer.optimizers',
# 'itcl_quantizer.quantizer',
# 'itcl_quantizer.quantizer.distributions',
# 'itcl_quantizer.quantizer.metrics',
# 'itcl_quantizer.tensor_extractor',
# 'itcl_quantizer.tensor_extractor.keras',
# 'itcl_quantizer.tensor_extractor.keras.layers',
# 'itcl_quantizer.util'
 ]
package_data = \
{'': ['*'],
 'itcl_quantizer': ['.venv/Lib/site-packages/tensorflow/include/external/cudnn_frontend_archive/_virtual_includes/cudnn_frontend/third_party/cudnn_frontend/include/contrib/nlohmann/json/*',
                    'models/keras/*',
                    'models/keras/regression/*',
                    'models/keras/regression/quant/*',
                    'models/keras/regression/x3/*',
                    'models/keras/wind_pred/*']}
install_requires = \
['matplotlib>=3.5.2,<4.0.0',
 'numpy>=1.23.1,<2.0.0',
 'scikit-learn>=1.1.1,<2.0.0',
 'scipy>=1.8.1,<2.0.0',
 'seaborn>=0.11.2,<0.12.0',
 'simanneal>=0.5.0,<0.6.0',
 'tensorflow>=2.9.0,<2.10.0']
setup_kwargs = {
    'name': 'itcl-quantizer',
    'version': '0.1.5',
    'description': 'Itcl Quantization Tool',
    'long_description': None,
    'author': 'ITCL',
    'author_email': None,
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<3.11',
}

setup(**setup_kwargs)

