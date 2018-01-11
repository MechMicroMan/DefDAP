    def floodFill(self, x, y, grainIndex):
        currentGrain = Grain(self)

        currentGrain.addPoint(self.quatArray[y, x], (x, y))

        edge = [(x, y)]
        grain = [(x, y)]

        self.grains[y, x] = grainIndex
        while edge:
            newedge = []

            for (x, y) in edge:
                moves = np.array([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

                movesIndexShift = 0
                if x <= 0:
                    moves = np.delete(moves, 1, 0)
                    movesIndexShift = 1
                elif x >= self.xDim - 1:
                    moves = np.delete(moves, 0, 0)
                    movesIndexShift = 1

                if y <= 0:
                    moves = np.delete(moves, 3 - movesIndexShift, 0)
                elif y >= self.yDim - 1:
                    moves = np.delete(moves, 2 - movesIndexShift, 0)

                for (s, t) in moves:
                    if self.grains[t, s] == 0:
                        currentGrain.addPoint(self.quatArray[t, s], (s, t))
                        newedge.append((s, t))
                        grain.append((s, t))
                        self.grains[t, s] = grainIndex
                    elif self.grains[t, s] == -1 and (s > x or t > y):
                        currentGrain.addPoint(self.quatArray[t, s], (s, t))
                        grain.append((s, t))
                        self.grains[t, s] = grainIndex

            if newedge == []:
                return currentGrain
            else:
                edge = newedge

    def floodFill(self, x, y, grainIndex):

        currentGrain.addPoint((x, y), self.max_shear[y + self.cropDists[1, 0], x + self.cropDists[0, 0]])

                        currentGrain.addPoint((s, t), self.max_shear[y + self.cropDists[1, 0], x + self.cropDists[0, 0]])

                        currentGrain.addPoint((s, t), self.max_shear[y + self.cropDists[1, 0], x + self.cropDists[0, 0]])


