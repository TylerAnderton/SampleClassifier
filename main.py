# run in the terminal with the command `pythonw main.py`

import drum_sample_classifier 

import wx
from wx.lib import newevent

MODEL_FILEPATH: str = 'model_training/trained_hypermodel'
classifier = drum_sample_classifier.DrumClassifier(MODEL_FILEPATH)

drop_event, EVT_DROP_EVENT = wx.lib.newevent.NewEvent()

class FileDropTarget(wx.FileDropTarget):
    def __init__(self, obj):
        wx.FileDropTarget.__init__(self)
        self.obj = obj

    def OnDropFiles(self, x, y, filename: list[str]):
        print(filename[0]+'\n')
        prediction: str = '\n' + classifier.make_and_return_prediction(filename[0])
        event = drop_event(data=prediction) 
        wx.PostEvent(self.obj, event) 
        return True

class Frame(wx.Frame):

    def __init__(self, parent, title):
        super(Frame, self).__init__(parent, title=title, size=(500, 600))
        self.InitUI()
        self.Center()

    def InitUI(self):

        # create elements
        panel = wx.Panel(self)

        drop_label = wx.StaticText(panel, size=(380, 30), label = 'Drop a .wav drum sample here', style=wx.ALIGN_CENTER)

        drop_place = wx.TextCtrl(panel, size=(380, 300))
        drop_place.SetDropTarget(FileDropTarget(self))
        self.Bind(EVT_DROP_EVENT, self.PredictionTextUpdate)

        pred_label = wx.StaticText(panel, size=(380, 30), label = 'Predictions will show here', style=wx.ALIGN_CENTER)

        self.pred_text = wx.TextCtrl(panel, size=(380, 60), style=wx.TE_MULTILINE|wx.TE_WORDWRAP|wx.TE_READONLY|wx.ALIGN_CENTER)

        # stylize
        label_font = wx.Font(wx.FontInfo(20).Bold())
        drop_label.SetFont(label_font)
        pred_label.SetFont(label_font)

        pred_font = wx.Font(wx.FontInfo(16))
        self.pred_text.SetFont(pred_font)

        # place elements
        vertical_box = wx.BoxSizer(wx.VERTICAL)

        vertical_box.Add((-1, 20))

        vertical_box.Add(drop_label, flag=wx.CENTER|wx.ALL, border=10)
        vertical_box.Add(drop_place, flag=wx.CENTER)

        vertical_box.Add((-1, 40))

        vertical_box.Add(pred_label, flag=wx.CENTER|wx.ALL, border=10)
        vertical_box.Add(self.pred_text, flag=wx.CENTER)

        panel.SetSizer(vertical_box)
        self.Show()

    def PredictionTextUpdate(self, event):
        self.pred_text.Clear()
        self.pred_text.write(event.data)

def main():
    app = wx.App()
    frame = Frame(None, title = 'Drum Sample Identifier')
    app.MainLoop()

if __name__ == '__main__':
    main()