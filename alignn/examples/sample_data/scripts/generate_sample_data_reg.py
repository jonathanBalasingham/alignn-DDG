"""Module to generate example dataset."""
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms

dft_3d = jdata("dft_3d")
prop = "dfpt_piezo_max_dij"  # dfpt_piezo_max_eij, bulk_modulus_kv, slme, mbj_bandgap, shear_modulus_gv, dfpt_piezo_max_dij
max_samples = 500000
f = open("id_prop.csv", "w")
count = 0
for i in dft_3d:
    atoms = Atoms.from_dict(i["atoms"])
    jid = i["jid"]
    poscar_name = "POSCAR-" + jid + ".vasp"
    target = i[prop]
    if target != "na":
        atoms.write_poscar(poscar_name)
        f.write("%s,%6f\n" % (poscar_name, target))
        count += 1
        if count == max_samples:
            break
f.close()
