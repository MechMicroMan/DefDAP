import numpy as np

from matplotlib.widgets import Button


class Map(object):
    def setHomogPoint(self):
        self.selPoint = None

        self.plotEulerMap()
        homogPoints = np.array(self.homogPoints)
        self.ax.scatter(x=homogPoints[:, 0], y=homogPoints[:, 1], c='y', s=60)

        btnAx = self.fig.add_axes([0.8, 0.0, 0.1, 0.07])
        Button(btnAx, 'Save point', color='0.85', hovercolor='0.95')

        # coonect click handler
        self.fig.canvas.mpl_connect('button_press_event', self.clickHomog)

    def clickHomog(self, event):
        if event.inaxes is not None:
            # clear current axis and redraw map
            self.ax.clear()
            self.plotEulerMap(updateCurrent=True)

            if event.inaxes is self.fig.axes[0]:
                # axis 0 then is a click on the map. Update selected point and plot
                self.selPoint = (int(event.xdata), int(event.ydata))
                self.ax.scatter(x=self.selPoint[0], y=self.selPoint[1], c='w', s=60, marker='x')

            elif (event.inaxes is self.fig.axes[1]) and (self.selPoint is not None):
                # axis 1 then is a click on the button. Add selected point to list
                self.homogPoints.append(self.selPoint)
                self.selPoint = None

            homogPoints = np.array(self.homogPoints)
            self.ax.scatter(x=homogPoints[:, 0], y=homogPoints[:, 1], c='y', s=60)

            self.fig.canvas.draw()
