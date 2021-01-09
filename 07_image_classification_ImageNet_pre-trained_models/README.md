<div style="text-align:center"><a href="https://www.youtube.com/watch?v=m3A1z7DFLXQ"><img src="https://i.imgur.com/jshMjtR.jpg" title="source: imgur.com" /></a></div>

<h3>Keras – Classificação de imagem usando modelos pré-treinados no ImageNet e TensorFlow 2.0</h3>

<p>Classificação de imagem usando modelos pré-treinados no ImageNet com Keras 2.3.0 e TensorFlow 2.0 como backend.</p>

<p>Os pesos são baixados automaticamente na primeira execução do código e ficam salvos em <b>/home/user/.keras/models</b>.<p/>


Corrigir erro:
<p><b>...</p></b>
<p><b>Function call stack:</p></b>
<p><b>keras_scratch_graph</p></b>

<p>Adicionar no código</p>

```
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

<b>Referências</b>
<ul>
  <li>https://keras.io/api/applications/</li>
  <li>https://www.tensorflow.org/api_docs/python/tf/keras/applications</li>
  <li>https://stackoverflow.com/questions/41910617/keras-valueerror-decode-predictions-expects-a-batch-of-predictions</li>
  <li>https://stackoverflow.com/questions/57062456/function-call-stack-keras-scratch-graph-error</li>
  <li>https://www.learnopencv.com/keras-tutorial-using-pre-trained-imagenet-models/</li>
  <li>https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/</li>
</ul>
