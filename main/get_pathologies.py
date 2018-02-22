def get_pathologies(filepath):
    """ Given a filepath, exports the pathologies."""
    data = pd.read_csv(filepath)
    data = data.values

    return data
