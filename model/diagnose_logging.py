# log
import logging

class Logger():
	def __init__(self, logName):
		# create a logger
		self.logger = logging.getLogger(logName)

		# if logger.handlers list is null, add one, if not, write the log directly
		if not self.logger.handlers:
			self.logger.setLevel(logging.INFO)

			# create a handler
			fh = logging.FileHandler('log.log')
			fh.setLevel(logging.INFO)

			# create a handler to output to console
			ch = logging.StreamHandler()
			ch.setLevel(logging.INFO)

			# define the format of handler output
			formatter = logging.Formatter('%(levelname)s:%(asctime)s -%(name)s -%(message)s')
			fh.setFormatter(formatter)
			ch.setFormatter(formatter)

			# add handler processer to logger
			self.logger.addHandler(fh)
			self.logger.addHandler(ch)

	def getlog(self):
		return self.logger
