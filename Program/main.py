import numpy as np
import matplotlib.pyplot as plt
import string
import sys
import matplotlib

from Windows import Ui_MainWindow
from Hopfield import Hopfield
from result_visible import drawPlot

from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
matplotlib.use('Qt5Agg')

class MyMainWindow(QMainWindow, Ui_MainWindow):

	def __init__(self, parent=None):
		super(MyMainWindow, self).__init__(parent)
		self.setupUi(self)
		self.inputfilepath_btn.clicked.connect(self.choosetrainfileDialog)
		self.inputfilepath_btn_2.clicked.connect(self.choosetestfileDialog)
		self.startButton.clicked.connect(self.start)
		self.canva = drawPlot()
		self.plot_layout.addWidget(self.canva)


	def data_preprocessing(self,filepath):
		x_train = []
		with open(filepath, 'r') as fread:
			tmp_array = []
			for line in fread.readlines():
				if line.strip('\n') == '':
					x_train.append(tmp_array)
					tmp_array = []
					continue
				line = line.strip('\n')
				line = list(line)
				for i in range(len(line)):
					if line[i] == ' ':
						line[i] = -1
					else:
						line[i] = 1
				tmp_array.append(line)
			x_train.append(tmp_array)
			tmp_array = []
		x_train = np.array(x_train)
		return x_train

	def choosetrainfileDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
												  "All Files (*);;Text Files (*.txt)", options=options)
		if filename:
			self.trainfilepath_edit.setText(filename)
		else:
			self.trainfilepath_edit.setText("")

	def choosetestfileDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
												  "All Files (*);;Text Files (*.txt)", options=options)
		if filename:
			self.testfilepath_edit.setText(filename)
		else:
			self.testfilepath_edit.setText("")

	def start(self):
		train_file = self.trainfilepath_edit.text()
		test_file = self.testfilepath_edit.text()
		x_train = self.data_preprocessing(train_file)
		x_test = self.data_preprocessing(test_file)

		c, h, w = x_train.shape
		
		hopfield = Hopfield()
		hopfield.train(x_train.reshape(-1,h*w))

		self.canva.drawresult(x_train, x_test, hopfield)


if __name__ == '__main__':

	# 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
	app = QApplication(sys.argv)
	# 初始化
	myWin = MyMainWindow()

	# 将窗口控件显示在屏幕上
	myWin.show()
	# 程序运行，sys.exit方法确保程序完整退出。
	sys.exit(app.exec_())

	# train_file = "D:/Project/NNHomework/NN_HW3/Dataset/Bonus_Training.txt"
	# test_file = "D:/Project/NNHomework/NN_HW3/Dataset/Bonus_Testing.txt"
	# x_train = data_preprocessing(train_file)
	# x_test = data_preprocessing(test_file)

	# c, h, w = x_train.shape

	# x_train = x_train.reshape(-1,h*w)
	# x_test = x_test.reshape(-1,h*w)
	
	# hopfield = Hopfield()
	# hopfield.train(x_train)
	# figure, axes = plt.subplots(3, len(x_test))
	# axes[0][0].set_ylabel("Train")
	# axes[1][0].set_ylabel("Test: Before Predict")
	# axes[2][0].set_ylabel("Test: After Predict")
	# for i in range(len(x_test)):
	# 	axes[0][i].imshow(np.array(x_train[i]).reshape((h,w)))
	# 	axes[1][i].imshow(np.array(x_test[i]).reshape((h,w)))
	# 	axes[2][i].imshow(np.array(hopfield.predict(x_test[i])).reshape((h,w)))
	# plt.show()