import tensorflow as tf

class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1, aggregation='mean'):
        super(GraphConvLayer, self).__init__()
        
        self.units = units
        self.dropout_rate = dropout_rate
        self.aggregation = aggregation
        
    def build(self, inputs):
        # inputs = (node_embedding, edges, edge_weights)
        node_embedding, edges, edge_weights = inputs
        
        self.message_passing = tf.keras.Sequential([tf.keras.layers.BatchNormalization(), 
                                                   tf.keras.layers.Dropout(rate=self.dropout_rate),
                                                   tf.keras.layers.Dense(units=self.units, activation='gelu')])
        
        self.embedding_update = tf.keras.Sequential([tf.keras.layers.BatchNormalization(), 
                                                    tf.keras.layers.Dropout(rate=self.dropout_rate),
                                                    tf.keras.layers.Dense(units=self.units, activation='gelu')])
        

    def call(self, inputs, training=False):
        node_embedding, edges, edge_weights = inputs
        node_indices, neighbour_indices = edges[0], edges[1]
        neighbour_embeddings = tf.gather(node_embedding, neighbour_indices)
        
        # 1 message passing
        messages = self.message_passing(neighbour_embeddings, training=training)
        weighted_messages = messages * tf.expand_dims(edge_weights, -1)
        
        # 2 aggregation
        num_nodes = tf.shape(node_embedding)[0]
        if self.aggregation=='mean':
            aggregated_message = tf.math.unsorted_segment_mean(
                weighted_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError("%s aggregation type not implemented."%self.aggregation)
        
        # 3 update
        h = node_embedding + aggregated_message
        h_new = self.embedding_update(h)
        h_new = tf.nn.l2_normalize(h_new, axis=-1)
        
        return h_new  