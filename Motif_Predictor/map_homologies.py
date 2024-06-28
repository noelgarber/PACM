import urllib.request
import tarfile
import io
import os
import pickle
from biomart import BiomartServer, BiomartException

# Define common mismatches
common_taxonomy_mismatches = {"Magnaporthe poae genes (Mag_poae_ATCC_64411_V1)": 148304,
                              "Gaeumannomyces tritici R3-111a-1 genes (Gae_graminis_V2)": 644352,
                              "Ashbya gossypii genes (ASM9102v1)": 33169,
                              "Candida auris genes (GCA002759435v2)": 498019,
                              "Magnaporthe oryzae genes (MG8)": 318829,
                              "Magnaporthe oryzae genes (GCA900474545v2)": 318829,
                              "Candida glabrata genes (GCA000002545v2)": 5478,
                              "Dioscorea rotundata genes (TDr96_F1_v2_PseudoChromosome)": 55577,
                              "Oryza sativa Indica Group genes (ASM465v1)": 39946,
                              "Pythium vexans genes (pve_scaffolds_v1)": 907947,
                              "Phytophthora parasitica genes (Phyt_para_P1569_V1)": 4792,
                              "Pythium irregulare genes (pir_scaffolds_v1)": 36331,
                              "Pig genes (Sscrofa11.1)": 9823,
                              "C.savignyi genes (CSAV 2.0)": 51511,
                              "Eastern brown snake genes (EBS10Xv2-PRI)": 8673,
                              "Goat genes (ARS1)": 9925,
                              "Opossum genes (ASM229v1)": 13616,
                              "Mouse Lemur genes (Mmur_3.0)": 30608,
                              "Javanese ricefish genes (OJAV_1.1)": 123683,
                              "Sailfin molly genes (P_latipinna-1.0)": 48699,
                              "Clown anemonefish genes (ASM2253959v1)": 80972,
                              "Algerian mouse genes (SPRET_EiJ_v1)": 10096,
                              "Chinese medaka genes (ASM858656v1)": 183150,
                              "Cow genes (ARS-UCD1.3)": 9913,
                              "Hagfish genes (Eburgeri_3.2)": 7764,
                              "Spiny chromis genes (ASM210954v1)": 80966,
                              "Goldfish genes (ASM336829v1)": 7957,
                              "Lamprey genes (Pmarinus_7.0)": 7757,
                              "Guppy genes (Guppy_female_1.0_MT)": 8081,
                              "Large yellow croaker genes (L_crocea_2.0)": 215358,
                              "Black snub-nosed monkey genes (ASM169854v1)": 61621,
                              "C.intestinalis genes (KH)": 7719,
                              "Midas cichlid genes (Midas_v5)": 61819,
                              "Giant panda genes (ASM200744v2)": 9646,
                              "Tuatara genes (ASM311381v1)": 8508,
                              "Zebra mbuna genes (M_zebra_UMD2a)": 106582,
                              "Mouse genes (GRCm39)": 10090,
                              "Rat genes (mRatBN7.2)": 10116,
                              "Turbot genes (ASM1334776v1)": 52904,
                              "Greater amberjack genes (Sdu_1.0)": 41447,
                              "Platyfish genes (X_maculatus-5.0-male)": 8083,
                              "Stickleback genes (GAculeatus_UGA_version5)": 69293,
                              "Tetraodon genes (TETRAODON 8.0)": 99883,
                              "Tiger tail seahorse genes (H_comes_QL1_v1)": 109280,
                              "Narwhal genes (NGI_Narwhal_1)": 40151,
                              "Sheepshead minnow genes (C_variegatus-1.0)": 28743,
                              "American black bear genes (ASM334442v1)": 9643}

# Download and extract the taxonomy dump
names_dmp_url = "ftp://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz"
cwd = os.getcwd()
def fetch_taxonomy_dump(url = names_dmp_url, save_folder = cwd):
    save_path = os.path.join(save_folder, "taxid_species_dict.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            taxid_species_dict = pickle.load(f)
    else:
        response = urllib.request.urlopen(url)
        with io.BytesIO(response.read()) as gzfile:
            with tarfile.open(fileobj=gzfile, mode="r:gz") as tar:
                names_dmp = None
                for member in tar.getmembers():
                    if member.name.endswith("names.dmp"):
                        names_dmp = tar.extractfile(member).read().decode("utf-8")
                        break
                if names_dmp is None:
                    raise ValueError("names.dmp not found in the archive")

        taxid_species_dict = {}
        for line in names_dmp.splitlines():
            parts = line.split("\t|\t")
            taxid = parts[0].strip()
            name = parts[1].strip()
            name_class = parts[3].rsplit("\t", 1)[0]
            if name_class == "scientific name" and len(name.split(" ")) == 2:
                taxid_species_dict[name] = int(taxid)

        with open(save_path, "wb") as f:
            pickle.dump(taxid_species_dict, f)

    return taxid_species_dict

def infer_taxid(species_name, species_prefix_dict, prefix_name_dict, dataset_descriptions, subdomain = "www"):
    # Infer TaxID by species name

    species_taxid = None
    species_taxids = species_prefix_dict.get(species_name)
    if species_taxids is not None:
        if len(species_taxids) == 1:
            species_taxid = species_taxids[0]
        else:
            species_full_names = prefix_name_dict[species_name]
            dataset_name = f"{species_name}_gene_ensembl" if subdomain == "www" else f"{species_name}_eg_gene"
            dataset_genus_species = " ".join(dataset_descriptions[dataset_name].split(" ")[:2])
            species_taxid = None
            for taxid, full_name in zip(species_taxids, species_full_names):
                if dataset_genus_species == full_name:
                    species_taxid = taxid
                    break

                # Catch cases where an abbreviated species is given, like GalGal or BosTau
                dataset_description = dataset_descriptions[dataset_name]
                full_name_elements = full_name.split(" ")
                tax_GenSpe = full_name_elements[0][:3] + full_name_elements[1][:3]
                GenSpe_within = tax_GenSpe.lower() in dataset_description.lower()
                tax_Gen_Spe = full_name_elements[0][:3] + "_" + full_name_elements[1][:3]
                Gen_Spe_within = tax_Gen_Spe.lower() in dataset_description.lower()
                gen_spe_found = GenSpe_within or Gen_Spe_within
                if gen_spe_found:
                    print(f"\tInferred that {dataset_description} is {full_name}")
                    species_taxid = taxid
                    break

                # Catch cases where species is separated by underscore
                genus_species_underscore = "_".join(full_name.split(" "))
                if genus_species_underscore.lower() in dataset_descriptions[dataset_name].lower():
                    species_taxid = taxid
                    break

                # For persistent cases where common mismatches occur, a useful dict is provided
                species_taxid = common_taxonomy_mismatches.get(dataset_descriptions[dataset_name])

            if species_taxid is None:
                print(f"\tAmbiguity found for {species_name} ({dataset_descriptions[dataset_name]}); "
                      f"please select from the following taxids: ")
                for taxid, full_name in zip(species_taxids, species_full_names):
                    print(f"\t\t{taxid} ({full_name})")
                while True:
                    species_taxid = input(f"\tEnter taxid:  ")
                    try:
                        species_taxid = int(species_taxid)
                        break
                    except:
                        print(f"\t\tNot an integer; please try again.")

    return species_taxid

def fetch_dataset_names(url = names_dmp_url, save_folder = cwd):
    save_path = os.path.join(save_folder, "datasets_dict.pkl")

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            datasets_dict = pickle.load(f)

    else:
        taxid_species_dict = fetch_taxonomy_dump(url, save_folder)

        # Get initial dict of genus letter + species --> taxid
        species_prefix_dict = {}
        prefix_name_dict = {}
        for key, taxid in taxid_species_dict.items():
            elements = key.split(" ")
            if len(elements) == 2:
                prefix = elements[0][0] + elements[1]
                prefix = prefix.lower()

                if species_prefix_dict.get(prefix) is None:
                    species_prefix_dict[prefix] = [taxid]
                else:
                    species_prefix_dict[prefix].append(taxid)

                if prefix_name_dict.get(prefix) is None:
                    prefix_name_dict[prefix] = [key]
                else:
                    prefix_name_dict[prefix].append(key)

        biomart_urls = ["http://fungi.ensembl.org/biomart",
                        "http://plants.ensembl.org/biomart",
                        "http://protists.ensembl.org/biomart",
                        "http://www.ensembl.org/biomart"] # main one must be last, to preferentially use when species are present in more than one db

        datasets_dict = {}
        for url in biomart_urls:
            subdomain = url.split("//")[1].split(".")[0]
            datasets_dict[url] = {}
            server = BiomartServer(url)
            datasets = list(server.datasets.keys())
            dataset_descriptions = {str(key): str(value) for key, value in server.datasets.items()}
            gspecies_list = [dataset.split("_", 1)[0] for dataset in datasets]
            gspecies_list = list(set(gspecies_list))

            if any(["gene_ensembl" in dataset for dataset in datasets]):
                for i, species_name in enumerate(gspecies_list):
                    species_taxid = infer_taxid(species_name, species_prefix_dict, prefix_name_dict, dataset_descriptions, subdomain)
                    datasets_dict[url][species_taxid] = (f"{species_name}_gene_ensembl", f"{species_name}_genomic_sequence")
            elif any(["eg_gene" in dataset for dataset in datasets]):
                for i, species_name in enumerate(gspecies_list):
                    species_taxid = infer_taxid(species_name, species_prefix_dict, prefix_name_dict, dataset_descriptions, subdomain)
                    datasets_dict[url][species_taxid] = (f"{species_name}_eg_gene", f"{species_name}_eg_genomic_sequence")
            else:
                raise Exception("gene_ensembl and eg_gene datasets were not found in the biomart server")

        with open(save_path, "wb") as f:
            pickle.dump(datasets_dict, f)

    return datasets_dict

def parse_biomart_response(response, invert_order = False):
    # Helper function for parsing BioMart responses containing host and homolog gene IDs

    homologs = {}
    for line in response.iter_lines():
        line = line.decode("utf-8")
        columns = line.split("\t")

        if len(columns) == 2:
            if not invert_order:
                first_gene_id, second_gene_id = columns
            else:
                second_gene_id, first_gene_id = columns

            if first_gene_id and second_gene_id:
                if homologs.get(first_gene_id) is None:
                    homologs[first_gene_id] = [second_gene_id]
                else:
                    homologs[first_gene_id].append(second_gene_id)

    return homologs

def map_homologies(reference_taxid = 9606, target_taxids = (10090, 10116, 7955, 7227, 6239, 4932, 4896, 3702),
                   save_folder = cwd):
    # Main function for creating a dictionary of dictionaries of reference and homolog genes for given target taxids

    save_path = os.path.join(save_folder, "target_dicts.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            target_dicts = pickle.load(f)
    else:
        # Find datasets for reference and targets, and the biomart urls they are found within
        datasets_dict = fetch_dataset_names()
        reference_datasets, reference_url = None, None
        for url, subdomain_dict in datasets_dict.items():
            reference_datasets = subdomain_dict.get(reference_taxid)
            if reference_datasets is not None:
                reference_url = url
                break

        targets_datasets = {}
        for target_taxid in target_taxids:
            for url, subdomain_dict in datasets_dict.items():
                target_datasets = subdomain_dict.get(target_taxid)
                if target_datasets is not None:
                    target_url = url
                    targets_datasets[target_taxid] = (target_datasets, target_url)
                    break

        if reference_datasets is None:
            raise Exception(f"BioMart dataset could not be found for reference taxid {reference_taxid}")
        if all(target_datasets is None for target_datasets in targets_datasets):
            raise Exception(f"BioMart datasets could not be found for any of the target taxids: {target_taxids}")

        # Query BioMart reference with target species homologs
        reference_species_name = reference_datasets[0].split("_", 1)[0]
        server = BiomartServer(reference_url)
        reference_subdomain = reference_url.split("//", 1)[1].split(".", 1)[0]
        ensembl = server.datasets[reference_datasets[0]]
        target_dicts = {}
        for target_taxid in target_taxids:
            print(f"Querying BioMart for target taxid {target_taxid}...")
            target_datasets, target_url = targets_datasets[target_taxid]

            target_subdomain = target_url.split("//", 1)[1].split(".", 1)[0]
            target_species = target_datasets[0].split("_", 1)[0]

            # Define attributes for querying reference dataset for target species
            if reference_subdomain == "www":
                reference_attributes = ["ensembl_gene_id", f"{target_species}_homolog_ensembl_gene"] # different naming in main db
            else:
                reference_attributes = ["ensembl_gene_id", f"{target_species}_eg_homolog_ensembl_gene"]

            # Query the Ensembl database for this reference/target pair
            try:
                reference_response = ensembl.search({"filters": {}, "attributes": reference_attributes})
                target_response = None
            except BiomartException as e:
                print(f"\tTarget species {target_species} was not found in reference dataset {reference_datasets[0]}; "
                      f"attempting in reverse direction...")
                reference_response = None

                # Define attributes for querying target dataset for reference species
                if target_subdomain == "www":
                    target_attributes = ["ensembl_gene_id", f"{reference_species_name}_homolog_ensembl_gene"]
                else:
                    target_attributes = ["ensembl_gene_id", f"{reference_species_name}_eg_homolog_ensembl_gene"]

                # Query the Ensembl database for this target/reference pair
                server = BiomartServer(target_url)
                ensembl = server.datasets[target_datasets[0]]
                try:
                    target_response = ensembl.search({"filters": {}, "attributes": target_attributes})
                except BiomartException as e:
                    target_response = None
                    print(f"\tReference species {reference_species_name} was not found in "
                          f"target dataset {target_datasets[0]} either; adding {target_species} to exceptions list")

            if reference_response is not None:
                reference_target_homologs = parse_biomart_response(reference_response)
            elif target_response is not None:
                reference_target_homologs = parse_biomart_response(target_response, invert_order=True)
            else:
                reference_target_homologs = None

            target_dicts[target_taxid] = reference_target_homologs

        with open(save_path, "wb") as f:
            pickle.dump(target_dicts, f)

    return target_dicts

if __name__ == "__main__":
    target_dicts = map_homologies()