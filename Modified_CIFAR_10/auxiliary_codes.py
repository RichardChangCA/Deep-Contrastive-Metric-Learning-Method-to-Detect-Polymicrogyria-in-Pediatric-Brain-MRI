# focal loss
# in train_step and valid_step
bce = tf.keras.losses.BinaryFocalCrossentropy()

# weighted BCE loss
loss = tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=predictions, pos_weight=tf.constant(5.))
loss = np.average(loss) # only in valid_step

c = np.ones(latent_representation_dim)
c = np.zeros(latent_representation_dim)
c = np.random.uniform(0,1,latent_representation_dim)
