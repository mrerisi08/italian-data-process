import lightgbm as lgb
import numpy as np
import shap
import pandas as pd
import re
from sklearn.metrics import roc_auc_score as auc
import os
import matplotlib.pyplot as plt
import random
import time
import requests
import itertools
import pickle as pkl

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42)

df = pd.read_csv("../../Case I 6-12 binary ft >10 positives and normalized.csv", na_values=np.nan)
df = df.drop(["Unnamed: 0", "GLYCATED HEMOGLOBIN"], axis=1)
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '',
                                        x))  # no clue why or what this does https://stackoverflow.com/questions/60582050/lightgbmerror-do-not-support-special-json-characters-in-feature-name-the-same
df.drop("LABEL", axis=1).to_csv("lgbm_dumbed_down_data.csv", index=False)
BINARY_FT = ['HELICOBACTERPYLORIINFECTION', 'SHINGLES', 'CHRONICHEPATITISC3rd', 'INFECTIOUSMONONUCLEOSIS', 'KSTOMACH',
             'KCOLON', 'KPULMONARY', 'CUTANEOUSMELANOMA', 'KUDDLE', 'KPROSTATE', 'KPROSTATE_1myedit', 'KBLADDER',
             'KRIGHTKIDNEY', "METFORMIN", 'Lymphoma', 'COLONPOLYPS', 'GOITER', 'THYROIDNODULE', 'NONTOXICMULTINODULARGOITER',
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
             'LORAZEPAM', 'MACROGOLASSOCIAZIONI', 'MAGALDRATO', 'MESALAZINA', 'METILPREDNISOLONE',
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
             'Gender'] # I REMOVED METFORMIN HERE
datadf = pd.read_csv("lgbm_dumbed_down_data.csv")
X = datadf.to_numpy()
y = df["LABEL"].to_numpy()


p = np.random.permutation(len(y))
X = X[p]
y = y[p]

K_FOLDS = 5
fold_size = int(len(X) / K_FOLDS)
params = [{'objective':"binary", 'metric':"auc",  'verbosity':-1,
           'num_threads':9, 'eta':0.1, 'max_leaf':10,
           'min_data':5, 'feature_fraction':0.25, 'reg_alpha':1,
           'lambda':0, 'min_split_gain':0, 'max_bin':255,"seed":64}]

for param in params:
    all_preds = []
    CONTRIB = []
    expected_values = []
    for fold in range(K_FOLDS):
        start = fold * fold_size
        end = (fold + 1) * fold_size
        # print(fold, start, end)
        if fold != 4:
            X_train = [*X[:start], *X[end:]]
            y_train = [*y[:start], *y[end:]]
            X_test = X[start:end]
            y_test = y[start:end]
        else:
            X_train = X[:start]
            y_train = y[:start]
            X_test = X[start:]
            y_test = y[start:]


        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print(X_train.shape)
        print(y_train.shape)
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(datadf.columns),
                                 categorical_feature=BINARY_FT)

        bst = lgb.train(param, train_data)

        preds = bst.predict(X_test)
        all_preds += list(preds)
        # print([round(float(x),4) for x in preds])
        explainer = shap.Explainer(bst)
        shap_values = explainer(X_test)
        CONTRIB.append(shap_values)
        print(fold, auc(y_test, preds))
        # with open(f'models (with metformin)/bst_mdl{fold}.pkl', 'wb') as file:
        #     pkl.dump(bst, file)
    auc_val = auc(y, all_preds)
    print(auc_val)

    combined_shap_values = np.concatenate([sv.values for sv in CONTRIB], axis=0)
    combined_base_values = np.concatenate([sv.base_values for sv in CONTRIB], axis=0)
    combined_data = np.concatenate([sv.data for sv in CONTRIB], axis=0)

    combined_shap_values_obj = shap.Explanation(values=combined_shap_values,
                                                base_values=combined_base_values,
                                                data=combined_data,
                                                feature_names=list(df.columns))

    shap.plots.beeswarm(combined_shap_values_obj, show=False, max_display=20)
    plt.savefig("lgbm-shap.png",bbox_inches="tight")

