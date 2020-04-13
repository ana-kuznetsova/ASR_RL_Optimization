class DataSet:
    def __init__(self, data, num_tasks):
        self.data = data
        self.scores = []
        self.num_tasks = num_tasks
        self.tasks = None
        
    def create_tasks(self):
        
        def index_marks(nrows, chunk_size):
            return range(chunk_size, math.ceil(nrows / chunk_size)*chunk_size, chunk_size)
        
        def split(dfm, chunk_size):
            indices = index_marks(dfm.shape[0], chunk_size)
            return np.split(dfm, indices)
        
        sorted_df = self.data.sort_values(by=['compression_scores'], ascending=False)['path'].values
        chunk_size = len(sorted_df)//self.num_tasks
        
        self.tasks = split(sorted_df, chunk_size)