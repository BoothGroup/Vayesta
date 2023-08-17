from vayesta.core.bath.bath import Bath

#

class RPA_Boson_Bath(Bath):
    def __init__(self, fragment, project_dmet_order=0, project_dmet_mode='full'):
        self.project_dmet_order = project_dmet_order
        self.project_dmet_mode = project_dmet_mode
        super().__init__(fragment)


    def make_target_orbitals(self):
        pass



