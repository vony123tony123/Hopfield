import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use('Qt5Agg')

class drawPlot(FigureCanvas):

	def __init__(self,width=10,height=10,dpi=100):
		# 第一步：創建一個創建Figure
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		# 第二步：在父類中激活Figure窗口
		super(drawPlot, self).__init__(self.fig)  # 此句必不可少，否則不能顯示圖形
		# 第四步：就是畫圖，可以在此類中畫，也可以在其它類中畫,最好是在別的地方作圖
		FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.points = list()

	def drawresult(self, x_train, x_test, hopfield):
		self.fig.clear(True)
		axes = self.fig.subplots(3, len(x_test))

		c, h, w = x_train.shape

		axes[0][0].set_ylabel("Train")
		axes[1][0].set_ylabel("Test: Before Predict")
		axes[2][0].set_ylabel("Test: After Predict")
		for i in range(len(x_test)):
			axes[0][i].imshow(x_train[i])
			axes[1][i].imshow(x_test[i])
			y = hopfield.predict(x_test[i].flatten())
			axes[2][i].imshow(np.array(y).reshape((h,w)))
		plt.show()
		self.draw()