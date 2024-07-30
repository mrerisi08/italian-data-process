import lightgbm as lgb
import numpy as np
import shap
import pandas as pd
import re
from sklearn.metrics import roc_auc_score as auc
import os
import random
import time
import requests
import itertools


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42)

def send_ifttt_notification(message):
    url = f"https://maker.ifttt.com/trigger/python_notif_tester/with/key/cUlA4Bn82wLJshLLMLQwBt"
    data = {"value1": message}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print(f"Failed to send notification: {response.status_code}, {response.text}")


df = pd.read_csv("../../Case I 6-12 binary ft >10 positives and normalized.csv", na_values=np.nan)
df = df.drop(["Unnamed: 0", "GLYCATED HEMOGLOBIN"], axis=1)
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '',
                                        x))  # no clue why or what this does https://stackoverflow.com/questions/60582050/lightgbmerror-do-not-support-special-json-characters-in-feature-name-the-same
#
#
# print(df.columns)
# print(df)
df.drop("LABEL", axis=1).to_csv("lgbm_dumbed_down_data.csv", index=False)

BINARY_FT = ['HELICOBACTERPYLORIINFECTION', 'SHINGLES', 'CHRONICHEPATITISC3rd', 'INFECTIOUSMONONUCLEOSIS', 'KSTOMACH',
             'KCOLON', 'KPULMONARY', 'CUTANEOUSMELANOMA', 'KUDDLE', 'KPROSTATE', 'KPROSTATE_1myedit', 'KBLADDER',
             'KRIGHTKIDNEY', 'Lymphoma', 'COLONPOLYPS', 'GOITER', 'THYROIDNODULE', 'NONTOXICMULTINODULARGOITER',
             'Gravesdisease', 'TOXICMULTINODULARGOITER', 'POSTSURGICALHYPOTHYROIDISM',
             'HYPOTHYROIDISMfromchronicthyroiditis', 'HASHIMOTOTHYROIDITIS', 'HYPERCHOLESTEROLEMIA', 'DYSLIPIDEMIA',
             'DYSLIPIDEMIAHYPERLIPIDEMIA', 'MONOCLONALGAMMAPATHYofuncertainsignificance', 'GOUT', 'GILBERTSYNDROME',
             'OBESITY', 'SIDEOPENINGANEMIA', 'SENILEDEMENTIA', 'ARTERIOSCLEROTICDEMENTIA', 'PSYCHOSIS',
             'ANXIETY_1myedit', 'ANXIOUSDEPRESSION', 'DEPRESSION', 'ALZHEIMERSDISEASE', 'PARKINSONSDISEASE', 'EPILEPSY',
             'MIGRAINE', 'LEFTCARPALTUNNELSYNDROME', 'GLAUCOMA', 'HEARINGIMPAIRMENT', 'HYPERTENSION',
             'ACUTEMYOCARDIALINFARCTION_2myedit', 'PREVIOUSMYOCARDIALINFARCTION', 'CHRONICMYOCARDIALISCHEMIA',
             'CHRONICCARDIACISCHEMIA', 'MITRALINSUFFICIENCY', 'AORTICSTENOSIS',
             'SUPRAVENTRICULARPAROXYSTICALTACHYCARDIA', 'ATRIALFIBRILLATION', 'ATRIALFIBRILLATION_1myedit',
             'HEARTFAILURE', 'CEREBRALSTROKECEREBRALARTERYOCCLUSION', 'TRANSIENTCEREBRALISCHEMIA', 'STROKE',
             'MULTINFARTUALENCEPHALOPATHY', 'ThoracicAORTADILATION', 'VARICOSEOFLOWERLIMBS', 'Bronchopneumonia',
             'INFLUENCE', 'CHRONICOBSTRUCTIVEBRONCHITISCOPDflaredup', 'Bronchialasthma',
             'COPDCHRONICOBSTRUCTIVEBRONCHITIS', 'GASTROESOPHAGEALREFLUXWITHGRADEIANDIIESOPHAGITISGERD',
             'REFLUXESOPHAGITIS1AND2GRADE', 'GASTROESOPHAGEALREFLUX', 'GASTRICULCER', 'DUODENALULCER', 'INGUINALHERNIA',
             'HIATALHERNIA', 'COLONDIVERTICULUS', 'CHRONICHEPATITISFROMHEPATITISC', 'GALLBLADSTONESTONES',
             'CHRONICRENALFAILURE', 'NEPHROLITHIASIS', 'BENIGNPROSTATICHYPERTROPHYtransvesicaladenomyomectomy',
             'Miscarriage', 'PSORIASIS', 'GENERALIZEDARTHROSIS', 'LEFTLUMBOSCIATALGIA', 'LOWBACKPAIN',
             'RIGHTSCIATALGIA', 'SCAPOLHOMERALPERIARTHRITIS', 'OSTEOPOROSIS', 'DIZZINESS', 'SLEEPAPNEA',
             'COUGHincaseofflu', 'RIGHTRENALCOLIC', 'URINARYINCONTINENCE', 'FRACTUREOFTHEDORSALVERTEBRAEsomaticD7',
             'FRACTUREOFLUMBARVERTEBRAE', 'FRACTUREOFRIBS', 'RIGHTHUMERUSFRACTUREcomposedofthehead',
             'FRACTUREOFTHELEFTWRIST', 'PERTROCHANTERICFEMURFRACTURE', 'FRACTUREOFTHELEFTFEMUR',
             'FOODALLERGYprobablytofish', 'NORMALPREGNANCY', 'PREGNANCY', 'HIPPROSTHESIS', 'LEFTKNEEPROSTHESIS',
             'LACKOFAFAMILYABLETOPROVIDECARE', 'PERMANENTBLADDERCATHETERFOLEY', 'SHAPEDDIAPER',
             'CATARACTSURGERYwithlensimplant', 'DUODENOSCOPY', 'COLONOSCOPY', 'REMOVALOFSKINLESION',
             'SKULLCTBENCEPHALON', 'XrayofthedentalarchesORTHOPANORAMIC', 'CERVICALSPINEXray', 'LUMBOSACRALSPINEXray',
             'BILATERALMAMMOGRAPHY', 'CHESTCTCONTRAST', 'SNHEMITORACERXFORRIBS', 'CHESTXRAY',
             'COMPLETEABDOMENCTCONTRAST', 'ABDOMENRXDIRECTABDOMEN', 'LEFTSHOULDER', 'LEFTWRIST', 'RXHIPRX', 'RXKNEERX',
             'LEFTFOOTXray', 'THYROIDECHOHEADANDNECKECHO', 'CARDIACECHOECOCARDIOGRAPHY', 'ECOUMMS',
             'ECODOPPLERSUPRAORTICTRUNKSECOCOLOR', 'ECHOKIDNEYSANDADRENALS', 'ECHOLOWERABDOMEN', 'FULLABDOMENECHO',
             'ECOCOLORDOPPLERLOWERLIMBSART', 'OBSTETRICULTRASOUNDFETOPLACENTARYULTRASOUND', 'ECOSURFACEFABRICS',
             'TRANSVAGINALECHO', 'MRIofthebrainandbrainstemWITHOUTCONTROL', 'MRILUMBOSACRALSPINE', 'MRIRIGHTKNEE',
             'LUMBARANDFEMORALDENSITOMETRYDEXA', 'VCARDIOLOGICAcontrol', 'VNEUROLOGICAL', 'UROFLOWOMETRY',
             'VGYNECOLOGICAL', 'SPIROMETRYBREATHINGTESTS', 'EFFORTECGCYCLEERGOMETER', 'DYNAMICECGHOLTER', 'ECG',
             'VSENOLOGICA', 'ALTGPTALANINEAMINOTRANSFERASES', 'PREALBUMIN', 'ALPHA1FETOPROTEINS', 'AMYLASE',
             'ASTGOTASPARTATEAMINOTRANSFERASE', 'BETA2MICROGLOBULIN', 'TOTALBILIRUBIN', 'TOTALANDFRACTIONALBILIRUBIN',
             'SOCCER', 'CALPROTECTINIMMUNOMETRICfeces', 'CHLORINE', 'COBALAMINVITB12', 'HDLCHOLESTEROL',
             'LDLCHOLESTEROL', 'TOTALCHOLESTEROL', 'CPKCREATINKINASECK', 'CREATININE', 'E2ESTRADIOL',
             'STOOLOCCULTBLOODON3SAMPLES', 'FERRITIN', 'IRONSIDEREMIA', 'FOLATES', 'FSHFOLLITROPIN',
             'ALKALINEPHOSPHATASE', 'PHOSPHORUS', 'GAMMAGTGAMMAGLUTAMILTRANSPS', 'GLYCEMIA',
             'CHORIONICGONADOTROPINfreebetaS', 'BENCEJONESURINE', 'LDHLACTATEDEHYDROGENASE', 'LIPASE', 'LHLUTEOTROPINS',
             'TOTALMAGNESIUM', 'MICROALBUMINURIA', 'BRAINNATRIURETICPEPTIDEBNPorNT', 'POTASSIUM', 'PRLPROLACTIN',
             'PROTIDOGRAMTOTALPROTEINS', 'SODIUM', 'TSHTHYROTROPIN', 'T4FREEFREETHYROXINE', 'TRANSFERRIN',
             'TRIGLYCERIDES', 'T3FREEFREETRIODOTYRONINE', 'URICEMIA', 'UREAAZOTEMIA', 'Completeurineexam', 'VITAMIND',
             'ANTICITRULLINEANTIBODIES', 'INDIRECTCOOMBSTEST', 'AbANTITRANSGLUTAMINASE', 'ANTITHYROPEROXIDASEAbAbTPO',
             'ANTINUCLEUSAbANA', 'ANTITHYROGLOBULINAbTG', 'CA125CARBOHYDRATEANTIGEN125', 'CA153CARBOHYDRATEANTIGEN',
             'CA199CARBOHYDRATEANTIGEN', 'CEAEMBRYONICCARCINOANTIGEN', 'PSAPROSTATICSPECIFICANTIGEN', 'PSAFREE',
             'BLOODCHROMEFORMULA', 'RHEUMATOIDFACTORRHEUMATEST', 'FIBRINOGEN', 'BLOODGROUPABOandRh',
             'IgGIMMUNOGLOBULINS', 'PCRCREACTIVEPROTEIN', 'RETICULOCYTES', 'PTPROTHROMBINTIMEINR',
             'PTTPARTIALTHROMBOPLASTINTIME', 'TROPONINI', 'VES', 'VAGINALswab', 'PHARYNGEALSWAB',
             'URINECULTUREURINECULTURE', 'TSHR', 'PSAR', 'TASOAbANTIANTISTREPTOLYSINO', 'TOXOPLASMAANTIBODYTotalEIA',
             'TREPONEMAPALLIDUMTPHAquantitative', 'CYTOMEGALOVIRUSAbANTIBODIESfc', 'HEPATITISBHBsAgAbHBV',
             'HEPATITISBHBsAgAgHBVAUANTIGEN', 'HEPATITISCRNAPCRHCV', 'HEPATITISCAbHCV', 'HERPESVIRUSAbfc',
             'ANTIHIVAb12', 'RUBELLAVIRUSANTIBODYRUBEOTEST', 'PAPTESTVAGINALCYTOLOGICAL',
             'HISTOPATHOLOGYExcisionalSKINBIOPSY', 'EMGELECTROMIGRAPHY', 'PHYSIOKINESTHERAPY', 'MAGNETICTHERAPY',
             'INTELLECTUALIMPAIRMENTTEST', 'VPSYCHIATRICPSYCHIATRICINTERVIEW', 'VOPHELISTICS',
             'FIELDOFVISIONCAMPIMETRY', 'OPTICALCOHERENCETOMOGRAPHY', 'TONEAUDIOMETRICEXAM', 'WOUNDDRESSING',
             'INHALATIONTREATMENTS', 'URINECOLLECTIONBAGS', 'HEIGHT', 'WEIGHT', 'BMIBODYMASSINDEX',
             'CLEARANCECREATININECOCKROFT', 'GFRestimatedwithMDRDformula', 'ACETILCISTEINA', 'ACIDOACETILSALICILICO',
             'ACIDOALGINICO', 'ACIDOFOLICO', 'ACIDOURSODESOSSICOLICO', 'ALFUZOSINA', 'ALLOPURINOLO', 'ALPRAZOLAM',
             'ALTRESOSTANZEXTRATTDIEMORROIDIERAGADIANALIUTOPICO', 'AMIODARONE', 'AMITRIPTILINA', 'AMLODIPINA',
             'AMOXICILLINA', 'AMOXICILLINAEDINIBITORIENZIMATICI', 'APPARATOMUSCOLOSCHELETRICO', 'ATENOLOLO',
             'ATORVASTATINA', 'AZITROMICINA', 'BECLOMETASONE', 'BETAISTINA', 'BETAMETASONE', 'BIFOSFONATI',
             'BIMATOPROST', 'BISOPROLOLO', 'BRINZOLAMIDE', 'BROMAZEPAM', 'CALCIOASSOCIAZIONICONVITAMINADEOALTRIFARMACI',
             'CANRENONE', 'CARBOCISTEINA', 'CARVEDILOLO', 'CEFIXIMA', 'CEFPODOXIMA', 'CEFTRIAXONE', 'CETIRIZINA',
             'CIANOCOBALAMINA', 'CIPROFLOXACINA', 'CLARITROMICINA', 'CLOPIDOGREL', 'CLORTETRACICLINA',
             'CODEINAEPARACETAMOLO', 'COLECALCIFEROLO', 'DERIVATIBENZODIAZEPINICI', 'DESAMETASONE',
             'DESAMETASONEEDANTIMICROBICI', 'DICLOFENAC', 'DIGOSSINA', 'DIOSMINAASSOCIAZIONI', 'DOXAZOSIN',
             'DUTASTERIDE', 'ELETTROLITI', 'ENOXAPARINA', 'EPERISONE', 'ESCITALOPRAM', 'ESOMEPRAZOLO', 'ETORICOXIB',
             'FERMENTILATTICI', 'FERROSOSOLFATO', 'FLUCONAZOLO', 'FLUNISOLIDE', 'FLUOXETINA', 'FORMOTEROLOEBUDESONIDE',
             'FOSFOMICINA', 'FUROSEMIDE', 'FUROSEMIDEEFARMACIRISPARMIATORIDIPOTASSIO', 'GLIMEPIRIDE', 'IBUPROFENE',
             'IDROCLOROTIAZIDEEFARMACIRISPARMIATORIDIPOTASSIO', 'IDROXOCOBALAMINA', 'INSULINGLARGINE', 'IRBESARTAN',
             'ISOSORBIDEMONONITRATO', 'ITRACONAZOLO', 'KETOPROFENE', 'LANSOPRAZOLO', 'LERCANIDIPINA',
             'LEVODOPAEDINIBITOREDELLADECARBOSSILASI', 'LEVOFLOXACINA', 'LEVOSULPIRIDE', 'LEVOTIROXINASODICA',
             'LORAZEPAM', 'MACROGOLASSOCIAZIONI', 'MAGALDRATO', 'MESALAZINA', 'METFORMIN', 'METILPREDNISOLONE',
             'METILPREDNISOLONEACEPONATO', 'METOPROLOLO', 'METRONIDAZOLO', 'MINERALIVITAMINEAMINOACIDIPROTEINE',
             'NEBIVOLOLO', 'NIMESULIDE', 'OMEGA3TRIGLICERIDIINCLUSIALTRIESTERIEACIDI', 'OMEPRAZOLO',
             'ORGANISMIPRODUTTORIDIACIDOLATTICO', 'OXAPROZINA', 'PANTOPRAZOLO', 'PARACETAMOLO', 'PAROXETINA',
             'POLIVALENTI', 'POTASSIOCLORURO', 'PREDNISONE', 'PREGABALIN', 'PROMAZINA', 'PRULIFLOXACINA', 'QUETIAPINA',
             'QUINAPRIL', 'RAMIPRIL', 'RAMIPRILEAMLODIPINA', 'RAMIPRILEDIURETICI', 'RIFAXIMINA', 'ROSUVASTATINA',
             'SALBUTAMOLO', 'SALMETEROLOEFLUTICASONE', 'SERTRALINA', 'SIMVASTATINA', 'TAMSULOSINA', 'TAPENTADOLO',
             'TIAMFENICOLO', 'TIMOLOLO', 'TIMOLOLOASSOCIAZIONI', 'TIOCOLCHICOSIDE', 'TRAZODONE', 'VALACICLOVIR',
             'VALSARTAN', 'VALSARTANEDIURETICI', 'VENLAFAXINA', 'WARFARIN', 'Circulatorysystemaffections',
             'ArterialhypertensionStIIandIII', 'Asthma', 'Chronicactivehepatitis', 'Glaucoma',
             'Congenitalandacquiredhypothyroidism', 'Psychosis', 'Malignantneoplasms', 'Hashimotosthyroiditis',
             'Circulatorysystemaffectionscardiopulmonary', 'Arterialhypertensionwithoutorgandamage', 'Civildisabled100',
             '100civiliandisabledwithaccompaniment', 'Civiliandisabledpeopleover23100',
             'Earlydiagnosisoftumorsmammographicbetween45and69yearsevery2years',
             'Earlycolorectaldiagnosiseveryoneaged45andoverevery5years', 'Incomeandageexemption',
             'Unemployedincomeexemption', 'Socialpensionincomeexemption',
             'Incomeexemptionfromregionalanticrisismeasures', 'Certificateissuingservices',
             'Injuredatworkorsufferingfromprofessionalillness', 'Responsiblematernityprotection', 'Pregnancyatrisk',
             'Gender']



datadf = pd.read_csv("lgbm_dumbed_down_data.csv")
X = datadf.to_numpy()
y = df["LABEL"].to_numpy()

p = np.random.permutation(len(y))
X = X[p]
y = y[p]

K_FOLDS = 5
fold_size = int(len(X) / K_FOLDS)

# B_A_T_AUC = -1
# scr = open("lgb_basic_data.csv",'r')
# scr = scr.readlines()[1:]
# all_time_aucs = []
# for a in scr:
#     a = a.split(",")[0]
#     a = np.float64(a)
#     if a > B_A_T_AUC:
#         B_A_T_AUC = a
# print(B_A_T_AUC)
# for some stupid reason param = {"num_leaves":31, "objective":"binary","metric":"auc"} is the best.


def grid_search_param():
    param_ranges = {"eta": [0.001, 0.01, 0.1, 0.5],
                    "max_leaf": [5, 10, 15, 20, 31, 50],
                    "min_data": [5, 10, 20, 50],
                    "feature_fraction": [0.1, 0.25, 0.5, 0.75, 1],
                    "reg_alpha": [0.0, 0.01, 1, 5, 25],
                    "lambda": [0.0, 0.01, 1, 5, 25],
                    "min_split_gain": [0, 50, 100, 250],
                    "verbosity": -1,
                    "max_bin": [100, 255, 500],
                    "objective": "binary",
                    "metric": "auc",
                    "num_threads": 9
                    }

    constant_hyperparams = {k: v for k, v in param_ranges.items() if not isinstance(v, list)}
    list_hyperparams = {k: v for k, v in param_ranges.items() if isinstance(v, list)}

    # Generate all combinations of list hyperparameters
    list_keys = list(list_hyperparams.keys())
    list_values = list(list_hyperparams.values())
    combinations = list(itertools.product(*list_values))
    final_hyperparams = []
    for combination in combinations:
        config = constant_hyperparams.copy()
        config.update(zip(list_keys, combination))
        final_hyperparams.append(config)
    return final_hyperparams

HYPERPARAMETERS = grid_search_param()


FOLD_ARRAYS = {}
for fold in range(4):
    start = fold * fold_size
    end = (fold + 1) * fold_size
    FOLD_ARRAYS[fold] = ([*X[:start], *X[end:]],[*y[:start], *y[end:]],X[start:end],y[start:end])

AUCs = []
ITERS = 0
total_amt_of_iterations = len(HYPERPARAMETERS)
last_tick = time.time()
B_A_T_AUC = -1
overall_start = time.time()
for param in HYPERPARAMETERS:
    ITERS += 1
    if ITERS % 100 == 0:
        file = open("iter.txt", "w")
        start_dist = time.time()-overall_start
        it = (f"{ITERS} / {total_amt_of_iterations}\n"
              f"{start_dist:.0f} seconds since start {start_dist/60:.2f} minutes, {start_dist/3600:.4f} hours\n"
              f"{time.time()-last_tick:.6f} seconds since last 10\n"
              f"{(time.time()-overall_start)/ITERS:.3f} average time per model (for {ITERS} models)\n"
              f"{B_A_T_AUC:.6f} best AUC overall achieved")
        file.write(it)
        file.close()
        last_tick = time.time()
    if ITERS % 500 == 0:
        send_ifttt_notification(f"{ITERS} / {total_amt_of_iterations} best AUC is {B_A_T_AUC}. Estim time remaining: {((time.time()-overall_start)/ITERS)*(total_amt_of_iterations-ITERS)/3600:.3f} hrs")

    all_preds = []

    for fold in range(K_FOLDS):
        X_train, y_train, X_test, y_test = FOLD_ARRAYS[fold]

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(datadf.columns),
                                 categorical_feature=BINARY_FT)

        bst = lgb.train(param, train_data)

        preds = bst.predict(X_test)
        all_preds += list(preds)

    auc_val = auc(y, all_preds)
    print(auc_val)
    AUCs.append(auc_val)
    with open("grid_search_lgbm.csv","a") as file:
        str = f"\n{auc_val}"
        for val in param:
            str = f"{str},{param[val]}"
        file.write(str)
    if auc_val > B_A_T_AUC:
        B_A_T_AUC = auc_val

print(AUCs)
# bst = lgb.train(param, train_data, 10)

# GOAL: Hyperparameter tune a light-gbm model to maximize AUC
# steps:
# TODO Train a first model with 2 goals, get an AUC value and get np.array with its prediction confidence
