from ikomia import dataprocess
import Detectron2_PanopticSegmentation_process as processMod
import Detectron2_PanopticSegmentation_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class Detectron2_PanopticSegmentation(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.Detectron2_PanopticSegmentationProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.Detectron2_PanopticSegmentationWidgetFactory()
