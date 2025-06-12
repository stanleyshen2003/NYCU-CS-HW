
class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def computeFeature(self, ii):
        return ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] - (ii[self.y+self.height][self.x]+ii[self.y][self.x+self.width])

    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)
    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)

class HaarFeature:
    def __init__(self, posRegions, negRegions):
        self.posRegions = posRegions
        self.negRegions = negRegions
        
    def computeFeature(self, ii):
        return sum([pos.computeFeature(ii) for pos in self.posRegions]) - sum([neg.computeFeature(ii) for neg in self.negRegions])
    
    def __str__(self):
        return "Haar feature (positive regions=%s, negative regions=%s)" % (str(self.posRegions), str(self.negRegions))
