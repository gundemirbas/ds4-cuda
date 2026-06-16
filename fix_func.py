import sys
with open('/home/xnaexor/ds4/ds4-cuda/src/ds4_safetensors.c', 'r') as f:
    text = f.read()

old = '''void *sst_sharded_model_tensor_data(sst_sharded_model *m, sst_tensor *t) {
    if (!m || !t) return NULL;
    /* Find the model that actually contains this tensor by name */
    for (uint64_t i = 0; i < m->n_models; i++) {
        sst_tensor *found = sst_model_find_tensor(m->models[i], t->name);
            return data;
        }
    }
    return NULL;
}'''

new = '''void *sst_sharded_model_tensor_data(sst_sharded_model *m, sst_tensor *t) {
    if (!m || !t) return NULL;
    /* Find the model that actually contains this tensor by name */
    for (uint64_t i = 0; i < m->n_models; i++) {
        sst_tensor *found = sst_model_find_tensor(m->models[i], t->name);
        if (found == t) {
            return sst_model_tensor_data(m->models[i], t);
        }
    }
    return NULL;
}'''

if old in text:
    text = text.replace(old, new, 1)
    with open('/home/xnaexor/ds4/ds4-cuda/src/ds4_safetensors.c', 'w') as f:
        f.write(text)
    print('Fixed')
else:
    print('Old text not found')
    idx = text.find('void *sst_sharded_model_tensor_data')
    if idx >= 0:
        print(repr(text[idx:idx+400]))
