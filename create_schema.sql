CREATE SCHEMA covidb;

CREATE  TABLE covidb.exposure_type ( 
	exposure_type_id     serial  NOT NULL ,
	exposure_type_name   varchar(100)   ,
	exposure_type_description varchar   ,
	exposure_type_total_units varchar   ,
	CONSTRAINT pk_exposure_type_exposure_type_id PRIMARY KEY ( exposure_type_id )
 );

COMMENT ON COLUMN covidb.exposure_type.exposure_type_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.exposure_type.exposure_type_name IS 'Name of exposure type.';

COMMENT ON COLUMN covidb.exposure_type.exposure_type_description IS 'Description of exposure.';

COMMENT ON COLUMN covidb.exposure_type.exposure_type_total_units IS 'Units determining how the exposure is cumulated (e.g. pack-years for smoking).';

CREATE  TABLE covidb.observation_type ( 
	observation_type_id  serial  NOT NULL ,
	observation_category smallint   ,
	observation_name     varchar  NOT NULL ,
	observation_units    varchar  NOT NULL ,
	observation_regexp   varchar   ,
	CONSTRAINT pk_observation_types_observation_id PRIMARY KEY ( observation_type_id )
 );

COMMENT ON COLUMN covidb.observation_type.observation_type_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.observation_type.observation_category IS 'Values:\n1 = Observation from clinical interaction only (e.g. capillary refill)\n2 = Observation derived from a score or rule (e.g. AVPU scale)\n3 = Observation from a measurement device (e.g. blood pressure)\n4 = Observation derived from a setting on a device (e.g. PEEP)';

COMMENT ON COLUMN covidb.observation_type.observation_name IS 'Name of observation (e.g. temperature, heart rate).';

COMMENT ON COLUMN covidb.observation_type.observation_units IS 'Units for this type of observation.';

COMMENT ON COLUMN covidb.observation_type.observation_regexp IS 'Regular expression used to validate observations.';

CREATE  TABLE covidb.patient_data ( 
	patient_uid          char(128)  NOT NULL ,
	patient_site_uid     char(128)  NOT NULL ,
	patient_site_code    varchar   ,
	patient_transfer_site_code date   ,
	patient_covid_status smallint   ,
	patient_birth_date   date   ,
	patient_sex          char(1)   ,
	patient_birth_country char(2)   ,
	patient_ethnicity    smallint   ,
	patient_pregnant     bool   ,
	patient_postal_district char(3)   ,
	patient_assisted_living smallint   ,
	patient_code_status  smallint   ,
	patient_death        bool   ,
	patient_death_cause  varchar   ,
	patient_last_fu_date date   ,
	CONSTRAINT pk_patient_data_patient_id PRIMARY KEY ( patient_uid )
 );

COMMENT ON TABLE covidb.patient_data IS 'Table for temporally invariant patient characteristics.';

COMMENT ON COLUMN covidb.patient_data.patient_uid IS 'Primary key of this table. Globally unique identifier for a patient with a given RAMQ number. Generated deterministically from RAMQ number via PBKDF2. Not traceable back to patient; follows NIST best practices for cryptographically secure storage of password keys.\n\nProcedure for generation: https://bit.ly/39ANAXa';

COMMENT ON COLUMN covidb.patient_data.patient_site_uid IS 'Per-site unique identifier, generated deterministically from unique local site ID via PBKDF2. \n\nProcedure for generation: https://bit.ly/39ANAXa';

COMMENT ON COLUMN covidb.patient_data.patient_site_code IS 'Standardized code for the site at which the data was acquired.\n\nReference codes: https://bit.ly/2Jyw7nN';

COMMENT ON COLUMN covidb.patient_data.patient_transfer_site_code IS 'Site code of the transferring institution, if the patient has been transferred from elsewhere.';

COMMENT ON COLUMN covidb.patient_data.patient_covid_status IS 'COVID status of the patient.\n1 = Positive at current site\n2 = Negative at current site\n3 = Positive at other site (transferred + patient)\n4 = Negative at other site (transferred - patient)\n5 = Test result pending\n6 = Not tested for COVID\n7 = Recovered patient';

COMMENT ON COLUMN covidb.patient_data.patient_birth_date IS 'Birth date of the patient. Should be replaced by patient_age, computed as per the most recent birth date, in de-identified data exports.';

COMMENT ON COLUMN covidb.patient_data.patient_sex IS 'Equals "F" if the patient''s is female, "M" if the patient''s sex is male, "X" if the patient''s sex is other than male or female.';

COMMENT ON COLUMN covidb.patient_data.patient_birth_country IS 'Country of origin, as 2-letter ISO 3166 code (e.g. CA).';

COMMENT ON COLUMN covidb.patient_data.patient_ethnicity IS 'Ethnicity of the patient, as defined by WHO CRF.\n1 = Arab\n2 = Black\n3 = East Asian\n4 = South Asian\n5 = West Asian\n6 = Latin American\n7 = White\n8 = Aboriginal / First Nations\n9 = Other\n10 = Unknown';

COMMENT ON COLUMN covidb.patient_data.patient_pregnant IS 'True if the patient was pregnant at the time of COVID testing.';

COMMENT ON COLUMN covidb.patient_data.patient_postal_district IS 'First 3 letters of postal code of the patient (e.g. H3V).';

COMMENT ON COLUMN covidb.patient_data.patient_assisted_living IS 'True if the patient lives in an assisted living facility (semi-autonomous residence or long-term care facility).';

COMMENT ON COLUMN covidb.patient_data.patient_code_status IS 'Level of intervention, where:\n1 = full code\n2 = intubation, but no code\n3 = medical therapy only\n4 = comfort care only';

COMMENT ON COLUMN covidb.patient_data.patient_death IS 'True if the patient has died.';

COMMENT ON COLUMN covidb.patient_data.patient_death_cause IS 'ICD code representing the cause of death.';

COMMENT ON COLUMN covidb.patient_data.patient_last_fu_date IS 'Date of last follow-up at which the mortality status of the patient was assessed.';

CREATE  TABLE covidb.pcr_data ( 
	patient_uid          char(128)  NOT NULL ,
	pcr_data_id          serial  NOT NULL ,
	pcr_name             varchar(100)   ,
	pcr_status           smallint   ,
	pcr_loinc_code       timestamptz   ,
	pcr_sample_time      timestamptz   ,
	pcr_sample_site      smallint   ,
	pcr_result_time      timestamptz   ,
	pcr_result_positive  bool   ,
	CONSTRAINT pk_pcr_data_id PRIMARY KEY ( pcr_data_id )
 );

COMMENT ON COLUMN covidb.pcr_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.pcr_data.pcr_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.pcr_data.pcr_name IS 'Name of the PCR test being performed (e.g. "SaRS-CoV-2 PCR").';

COMMENT ON COLUMN covidb.pcr_data.pcr_status IS 'Current status of the culture:\n1 = Resulted\n2 = Pending\n3 = Rejected\n4 = Cancelled\n5 = Other';

COMMENT ON COLUMN covidb.pcr_data.pcr_loinc_code IS 'LOINC code identifying the test kit used for PCR testing. If a non-commercial kit was used, the appropriate code is 94500-6.\n\nReference: https://bit.ly/2wQJfly';

COMMENT ON COLUMN covidb.pcr_data.pcr_sample_time IS 'Timestamp, with timezone, indicating the time at which the sample was drawn.';

COMMENT ON COLUMN covidb.pcr_data.pcr_sample_site IS 'Type of sample collected.\n1 = Nasopharyngeal swab\n2 = Oropharyngeal swab\n3 = Combined NP + OP swab\n4 = Expectorated sputum\n5 = Bronchial aspirate\n6 = Bronchoalveolar lavage\n7 = Feces/rectal swab\n8 = Urine specimen\n9 = Blood specimen\n10 = Unknown/unspecified';

COMMENT ON COLUMN covidb.pcr_data.pcr_result_time IS 'Timestamp, with timezone, indicating the time at which the PCR test was resulted.';

COMMENT ON COLUMN covidb.pcr_data.pcr_result_positive IS 'True if the result from the PCR test was positive.';

CREATE  TABLE covidb.signal_data ( 
	patient_uid          char(128)  NOT NULL ,
	signal_data_id       serial  NOT NULL ,
	signal_type          smallint   ,
	signal_num_channels  smallint   ,
	signal_sampling_freq smallint   ,
	signal_data_uri      varchar   ,
	signal_start_time    timestamptz   ,
	signal_end_time      timestamptz   ,
	CONSTRAINT pk_signal_data_id PRIMARY KEY ( signal_data_id )
 );

COMMENT ON TABLE covidb.signal_data IS 'Raw signal data (e.g. EKG, pulse oximeter, arterial line)';

COMMENT ON COLUMN covidb.signal_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.signal_data.signal_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.signal_data.signal_type IS 'Type of signal being recorded.\n1 = arterial blood pressure\n2 = pulmonary artery pressure\n3 = pulse oximetry\n4 = ventilator pressure\n5 = ventilator volume\n6 = waveform capnography\n7 = electrocardiogram';

COMMENT ON COLUMN covidb.signal_data.signal_num_channels IS 'The number of channels in the signal being recorded.';

COMMENT ON COLUMN covidb.signal_data.signal_sampling_freq IS 'Sampling frequency, in hertz.';

COMMENT ON COLUMN covidb.signal_data.signal_data_uri IS 'Unique resource identifier pointing to the file on disk where the signal is stored.';

COMMENT ON COLUMN covidb.signal_data.signal_start_time IS 'Timestamp, with timezone, indicating when the signal recording started.';

COMMENT ON COLUMN covidb.signal_data.signal_end_time IS 'Timestamp, with timezone, indicating when the signal recording ended.';

CREATE  TABLE covidb.symptom_data ( 
	patient_uid          char(128)  NOT NULL ,
	symptom_data_id      serial  NOT NULL ,
	symptom_snomed_code  varchar   ,
	symptom_name         varchar(100)   ,
	symptom_details      varchar   ,
	symptom_source       varchar   ,
	symptom_start_time   timestamptz   ,
	symptom_end_time     timestamptz   ,
	CONSTRAINT pk_symptom_data_symptom_data_id PRIMARY KEY ( symptom_data_id )
 );

COMMENT ON TABLE covidb.symptom_data IS 'Data for symptoms experienced by the patient.';

COMMENT ON COLUMN covidb.symptom_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.symptom_data.symptom_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.symptom_data.symptom_snomed_code IS 'SNOMED CT code representing the symptom.';

COMMENT ON COLUMN covidb.symptom_data.symptom_name IS 'Name of the symptom experienced.';

COMMENT ON COLUMN covidb.symptom_data.symptom_details IS 'Free text containing any additional details about symptoms experienced.';

COMMENT ON COLUMN covidb.symptom_data.symptom_source IS 'Data source that was used to collect information on this symptom.';

COMMENT ON COLUMN covidb.symptom_data.symptom_start_time IS 'Timestamp, with timezone, indicating time at which the symptoms started.';

COMMENT ON COLUMN covidb.symptom_data.symptom_end_time IS 'Timestamp, with timezone, indicating time at which the symptoms ended.';

CREATE  TABLE covidb.culture_data ( 
	patient_uid          char(128)  NOT NULL ,
	culture_data_id      integer  NOT NULL ,
	culture_type         smallint   ,
	culture_status       smallint   ,
	culture_loinc_code   varchar   ,
	culture_sample_site  smallint   ,
	culture_sample_time  timestamptz   ,
	culture_result_time  timestamptz   ,
	culture_gram_positive bool   ,
	culture_growth_positive bool   ,
	culture_result_organisms bool   ,
	CONSTRAINT pk_culture_data_id PRIMARY KEY ( culture_data_id )
 );

COMMENT ON COLUMN covidb.culture_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.culture_data.culture_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.culture_data.culture_type IS 'Type of culture obtained.\n1 = Bacterial culture (aerobic)\n2 = Bacterial culture (anaerobic)\n3 = Fungal culture\n4 = Mycobacterial culture';

COMMENT ON COLUMN covidb.culture_data.culture_status IS 'Current status of the culture:\n1 = Resulted\n2 = Pending\n3 = Rejected\n4 = Cancelled\n5 = Other';

COMMENT ON COLUMN covidb.culture_data.culture_loinc_code IS 'LOINC code of the culture being performed.';

COMMENT ON COLUMN covidb.culture_data.culture_sample_site IS 'Site from which the culture sample was collected.\n1 = Expectorated sputum\n2 = Endotracheal aspirate\n3 = Bronchoalveolar lavage\n4 = Pleural fluid\n5 = Ascites fluid\n6 = Blood culture\n7 = Urine culture\n8 = CSF culture\n9 = Other';

COMMENT ON COLUMN covidb.culture_data.culture_sample_time IS 'Timestamp, with time zone, of sampling.';

COMMENT ON COLUMN covidb.culture_data.culture_result_time IS 'Timestamp, with time zone, indicating the time at which the culture was resulted.';

COMMENT ON COLUMN covidb.culture_data.culture_gram_positive IS 'Equals true if the Gram stain is positive.';

COMMENT ON COLUMN covidb.culture_data.culture_growth_positive IS 'Equals true if the culture yielded growth of a pathogen in significant quantities (e.g. any quantity for blood, > 10^3 CFUs for LBA sputum)';

COMMENT ON COLUMN covidb.culture_data.culture_result_organisms IS 'Comma-separated list of organisms that have grown on the culture medium. Organisms should be stated with their full name (e.g. Staphylococcus aureus).';

CREATE  TABLE covidb.drug_data ( 
	patient_uid          char(128)  NOT NULL ,
	drug_data_id         serial  NOT NULL ,
	drug_name            varchar(100)   ,
	drug_din_code        varchar   ,
	drug_class           varchar   ,
	drug_infusion        bool   ,
	drug_dose            varchar   ,
	drug_rate            float8   ,
	drug_roa             varchar   ,
	drug_frequency       smallint   ,
	drug_start_time      timestamptz   ,
	drug_end_time        timestamptz   ,
	CONSTRAINT pk_drug_data_id PRIMARY KEY ( drug_data_id )
 );

COMMENT ON COLUMN covidb.drug_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.drug_data.drug_name IS 'Generic name of drug that was administered.';

COMMENT ON COLUMN covidb.drug_data.drug_din_code IS 'Standard Drug Identification Number (DIN).\n\nReference: https://bit.ly/2R8jPGC';

COMMENT ON COLUMN covidb.drug_data.drug_class IS 'Standard drug class.';

COMMENT ON COLUMN covidb.drug_data.drug_infusion IS 'True if the drug is administered via a continuous infusion.';

COMMENT ON COLUMN covidb.drug_data.drug_dose IS 'Dose of the drug administered at a time.';

COMMENT ON COLUMN covidb.drug_data.drug_rate IS 'Drug infusion rate, for continuous infusions.';

COMMENT ON COLUMN covidb.drug_data.drug_roa IS 'Drug route of administration short name, as defined by FDA standards (e.g. IV, SC, PO). \n\nReference: https://bit.ly/3dRlo5z';

COMMENT ON COLUMN covidb.drug_data.drug_frequency IS '(If drug_infusion is FALSE). Frequency of drug administration, in number of times per day.';

COMMENT ON COLUMN covidb.drug_data.drug_start_time IS 'Timezone, with timestamp, indicating the time at which the drug was started.';

COMMENT ON COLUMN covidb.drug_data.drug_end_time IS 'Timezone, with timestamp, indicating the time at which the drug was stopped.';

CREATE  TABLE covidb.episode_data ( 
	patient_uid          char(128)  NOT NULL ,
	episode_data_id      serial  NOT NULL ,
	episode_unit_type    smallint   ,
	episode_ctas         date   ,
	episode_start_time   timestamptz   ,
	episode_end_time     timestamptz   ,
	CONSTRAINT pk_episode_data_id PRIMARY KEY ( episode_data_id )
 );

COMMENT ON TABLE covidb.episode_data IS 'Start and end type of patient presence on unit, with accompanying metadata.';

COMMENT ON COLUMN covidb.episode_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.episode_data.episode_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.episode_data.episode_unit_type IS 'Type of unit:\n1 = Outpatient clinic\n2 = Emergency room\n3 = Hospital ward\n4 = High-dependency unit\n5 = Intensive care unit';

COMMENT ON COLUMN covidb.episode_data.episode_ctas IS 'CTAS (Canadian Triage and Acuity Scale) level for initial presentation in ER.\nLevel 1 = Resuscitation\nLevel 2 = Emergent\nLevel 3 = Urgent\nLevel 4 = Less urgent\nLevel 5 = Non urgent\n\nShould be blank unless episode_unit_type == 2.';

COMMENT ON COLUMN covidb.episode_data.episode_start_time IS 'Timestamp, with timezone, of when the episode started (e.g. triaged in the emergency room).';

COMMENT ON COLUMN covidb.episode_data.episode_end_time IS 'Timestamp, with timezone, of when the episode ended (e.g. discharged from the emergency room).';

CREATE  TABLE covidb.exposure_data ( 
	patient_uid          char(128)  NOT NULL ,
	exposure_data_id     integer  NOT NULL ,
	exposure_type_id     integer   ,
	exposure_total       timestamptz   ,
	exposure_details     varchar   ,
	exposure_start_time  timestamptz   ,
	exposure_end_time    timestamptz   ,
	CONSTRAINT pk_exposure_data_id PRIMARY KEY ( exposure_data_id )
 );

COMMENT ON TABLE covidb.exposure_data IS 'Exposure dates and types.';

COMMENT ON COLUMN covidb.exposure_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.exposure_data.exposure_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.exposure_data.exposure_type_id IS 'Identifier referencing the type of exposure.';

COMMENT ON COLUMN covidb.exposure_data.exposure_total IS 'Cumulative exposure, if quantifiable. E.g. number of pack years smoked.';

COMMENT ON COLUMN covidb.exposure_data.exposure_details IS 'Free text with additional details on exposure.';

COMMENT ON COLUMN covidb.exposure_data.exposure_start_time IS 'Timestamp, with time zone, indicating when the exposure occurred (for a one-time exposure), or started (for an prolonged exposure).';

COMMENT ON COLUMN covidb.exposure_data.exposure_end_time IS 'Timestamp, with time zone, indicating when the exposure ended.';

CREATE  TABLE covidb.imaging_data ( 
	patient_uid          char(128)  NOT NULL ,
	imaging_data_id      serial  NOT NULL ,
	imaging_accession_number varchar   ,
	imaging_modality     timestamptz   ,
	imaging_iv_contrast  bool   ,
	imaging_manufacturer varchar   ,
	imaging_acquisition_time timestamptz   ,
	imaging_report_time  timestamptz   ,
	imaging_report_text  varchar(250)   ,
	CONSTRAINT pk_study_data_id PRIMARY KEY ( imaging_data_id )
 );

COMMENT ON COLUMN covidb.imaging_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.imaging_data.imaging_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.imaging_data.imaging_accession_number IS 'Accession number of the imaging on the site''s local PACS database.';

COMMENT ON COLUMN covidb.imaging_data.imaging_modality IS 'Modality of the imaging that was performed, as defined by the DICOM "Modality" header (DICOM standard section C.7.3.1).\n\nExamples: CT, XR, US.\n\nReference: https://bit.ly/2X8Yv7Q';

COMMENT ON COLUMN covidb.imaging_data.imaging_iv_contrast IS 'True if intravenous contrast was given during the imaging study.';

COMMENT ON COLUMN covidb.imaging_data.imaging_manufacturer IS 'Manufacturer of the equipment that was used to acquire imaging, as defined by the DICOM "Manufacturer header."';

COMMENT ON COLUMN covidb.imaging_data.imaging_acquisition_time IS 'Timestamp, with timezone, representing the time at which the study was performed.';

COMMENT ON COLUMN covidb.imaging_data.imaging_report_time IS 'Timestamp, with timezone, indicating the time at which the imaging was reported.';

COMMENT ON COLUMN covidb.imaging_data.imaging_report_text IS 'De-identified textual report of the imaging.';

CREATE  TABLE covidb.intervention_data ( 
	patient_uid          char(128)  NOT NULL ,
	drug_data_id         integer   ,
	intervention_data_id serial  NOT NULL ,
	intervention_type_id smallint   ,
	intervention_details varchar(100)   ,
	intervention_start_time timestamptz   ,
	intervention_end_time timestamptz   ,
	CONSTRAINT pk_intervention_data_id PRIMARY KEY ( intervention_data_id ),
	CONSTRAINT unq_intervention_data_intervention_type_id UNIQUE ( intervention_type_id ) 
 );

COMMENT ON COLUMN covidb.intervention_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.intervention_data.intervention_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.intervention_data.intervention_type_id IS 'Identifier referencing the type of intervention.';

COMMENT ON COLUMN covidb.intervention_data.intervention_start_time IS 'Timestamp, with timezone, indicating the time at which the intervention was initiated.';

COMMENT ON COLUMN covidb.intervention_data.intervention_end_time IS 'Timestamp, with timezone, indicating the time at which the intervention was completed.';

CREATE  TABLE covidb.intervention_type ( 
	intervention_type_id serial  NOT NULL ,
	intervention_name    varchar   ,
	intervention_invasiveness serial   ,
	CONSTRAINT pk_respiratory_data_id PRIMARY KEY ( intervention_type_id )
 );

COMMENT ON COLUMN covidb.intervention_type.intervention_type_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.intervention_type.intervention_name IS 'Name of intervention. E.g. prone ventilation, inhaled nitric oxide, tracheostomy, extracorporeal support.';

COMMENT ON COLUMN covidb.intervention_type.intervention_invasiveness IS 'Level of invasiveness of the intervention type:\n1 = noninvasive intervention (e.g. proning)\n2 = drug administration\n3 = invasive bedside procedure (e.g. central line)\n4 = open surgical procedure (e.g. tracheostomy)';

CREATE  TABLE covidb.lab_data ( 
	patient_uid          char(128)  NOT NULL ,
	lab_data_id          serial  NOT NULL ,
	lab_name             varchar(100)   ,
	lab_loinc_code       varchar   ,
	lab_cerner_code      varchar   ,
	lab_units            varchar   ,
	lab_status           smallint   ,
	lab_sample_time      timestamptz   ,
	lab_sample_type      smallint   ,
	lab_result_time      timestamptz   ,
	lab_result_value     float8   ,
	CONSTRAINT pk_lab_data_lab_data_id PRIMARY KEY ( lab_data_id )
 );

COMMENT ON TABLE covidb.lab_data IS 'Contains data related to laboratory tests.';

COMMENT ON COLUMN covidb.lab_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.lab_data.lab_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.lab_data.lab_name IS 'Name of laboratory test. \n\nReference: https://bit.ly/2UGF5pp\n\nCould be in: { haemoglobin, WBC count, lymphocyte count, neutrophil count, hematocrit, platelets, APTT, PT, INR, ALT, total bilirubin, AST, glucose, urea, lactate, creatinine, sodium, potassium, procalcitonin, CRP, LDH, D-dimer, fibrinogen, ferritin, triglycerides, IL-6, CD4, CD8, CD4/CD8 ratio, NT-proBNP, troponin, hsT, hsTnI, TnT, TnI }';

COMMENT ON COLUMN covidb.lab_data.lab_loinc_code IS 'LOINC code for the laboratory test.';

COMMENT ON COLUMN covidb.lab_data.lab_cerner_code IS 'Code for laboratory test.\n\nReference (for CERNER): https://bit.ly/2UGF5pp';

COMMENT ON COLUMN covidb.lab_data.lab_units IS 'Reporting units for the lab test.\n\nCould be in: { g/L, g/dL, 10^9 cells/L, 10^3 cells/microL, cells/microL, seconds, U/L, micromol/L, mg/L, mg/dL, mmol/L, mg/dL, mEq/L }';

COMMENT ON COLUMN covidb.lab_data.lab_status IS 'Current status of the culture:\n1 = Resulted\n2 = Pending\n3 = Rejected\n4 = Cancelled\n5 = Other';

COMMENT ON COLUMN covidb.lab_data.lab_sample_time IS 'Timezone, with timestamp, indicating the time at which the laboratory test was sampled.';

COMMENT ON COLUMN covidb.lab_data.lab_sample_type IS 'Type of specimen sample.\n1 = Venous blood\n2 = Arterial blood\n3 = Capillary blood\n4 = Urine\n5 = Other\n6 = Unspecified';

COMMENT ON COLUMN covidb.lab_data.lab_result_time IS 'Timezone, with timestamp, indicating the time at which the laboratory test was resulted.';

COMMENT ON COLUMN covidb.lab_data.lab_result_value IS 'Value of the laboratory result.';

CREATE  TABLE covidb.observation_data ( 
	patient_uid          char(128)  NOT NULL ,
	observation_data_id  serial  NOT NULL ,
	observation_type_id  integer  NOT NULL ,
	observation_device   varchar   ,
	observation_start_time timestamptz   ,
	observation_end_time timestamptz   ,
	CONSTRAINT pk_clinical_data_id PRIMARY KEY ( observation_data_id )
 );

COMMENT ON COLUMN covidb.observation_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.observation_data.observation_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.observation_data.observation_device IS 'Device that was used to perform the observation (e.g. blood pressure cuff, arterial line).';

COMMENT ON COLUMN covidb.observation_data.observation_start_time IS 'Timestamp, with timezone, indicating the time at which the observation started.';

COMMENT ON COLUMN covidb.observation_data.observation_end_time IS 'Timestamp, with timezone, indicating the time at which the observation ended. Empty if the observation is made at a single point in time.';

CREATE  TABLE covidb.slice_data ( 
	patient_uid          char(128)  NOT NULL ,
	imaging_data_id      integer   ,
	slice_data_id        serial  NOT NULL ,
	slice_series_id      varchar   ,
	slice_data_uri       varchar   ,
	slice_view_position  varchar   ,
	slice_patient_position varchar   ,
	slice_image_orientation varchar   ,
	slice_image_position varchar   ,
	slice_window_center  varchar   ,
	slice_window_width   varchar   ,
	slice_pixel_spacing  varchar   ,
	slice_thickness      varchar   ,
	CONSTRAINT pk_dicom_data_id PRIMARY KEY ( slice_data_id )
 );

COMMENT ON TABLE covidb.slice_data IS 'Diagnostic imaging data in DICOM format (X-ray, CT, ultrasound).';

COMMENT ON COLUMN covidb.slice_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.slice_data.imaging_data_id IS 'Identifier that references the imaging study to which this slice belongs.';

COMMENT ON COLUMN covidb.slice_data.slice_data_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.slice_data.slice_series_id IS 'As defined by the DICOM SeriesInstanceUID header (0020,000E).';

COMMENT ON COLUMN covidb.slice_data.slice_data_uri IS 'Unique resource identifier pointing to the binary object representing the pixel data.';

COMMENT ON COLUMN covidb.slice_data.slice_view_position IS 'As defined by the DICOM ViewPosition header (DICOM standard, C.8.1.1)\n\nShould be in: { AP, PA, LL, RL, RLD, LLD, RLO, LLO }';

COMMENT ON COLUMN covidb.slice_data.slice_patient_position IS 'As defined by the DICOM PatientPosition header (DICOM standard C.7.3.1).\n\nShould be in: { HFP, HFS, HFDR, HFDL, FFDR, FFDL, FFP, FFS, LFP, LFS, RFP, RFS, AFDR, AFDL, PFDR, PFDL }';

COMMENT ON COLUMN covidb.slice_data.slice_image_orientation IS 'Specifies the direction cosines of the first row and the first column with respect to the patient, as defined by the DICOM ImageOrientation header (DICOM standard C.7.6.2).\n\nShould be: a string containing a comma-separated list of 6 floating point numbers.';

COMMENT ON COLUMN covidb.slice_data.slice_image_position IS 'Specifies the X, Y, and Z coordinates of the upper left hand corner of the image, as defined by the DICOM ImagePosition header (DICOM standard C.7.6.2)\n\nShould be: a string containing a comma-separated list of 3 floating point numbers.';

COMMENT ON COLUMN covidb.slice_data.slice_window_center IS 'As defined by the DICOM WindowCenter header.';

COMMENT ON COLUMN covidb.slice_data.slice_window_width IS 'As defined by the DICOM WindowWidth header.';

COMMENT ON COLUMN covidb.slice_data.slice_pixel_spacing IS 'For CT scans only. As defined in the PixelSpacing DICOM header.';

COMMENT ON COLUMN covidb.slice_data.slice_thickness IS 'For CT scans only. As defined in the PixelSpacing DICOM header.';

CREATE  TABLE covidb.diagnosis_data ( 
	patient_uid          char(128)  NOT NULL ,
	episode_data_id      integer   ,
	diagnosis_id         serial  NOT NULL ,
	diagnosis_type       varchar   ,
	diagnosis_name       varchar(100)   ,
	diagnosis_icd_code   varchar   ,
	diagnosis_source     varchar   ,
	CONSTRAINT pk_outcome_data_id PRIMARY KEY ( diagnosis_id )
 );

COMMENT ON COLUMN covidb.diagnosis_data.patient_uid IS 'Identifier referencing a specific individual in the patient_data table.';

COMMENT ON COLUMN covidb.diagnosis_data.episode_data_id IS 'Identifier referencing episode during which a diagnosis was made.';

COMMENT ON COLUMN covidb.diagnosis_data.diagnosis_id IS 'Primary key of this table; implemented as an auto-incrementing integer.';

COMMENT ON COLUMN covidb.diagnosis_data.diagnosis_type IS 'Type of diagnosis recorded.\n1 = comorbidity prior to admission (BDCP type 1)\n2 = comorbidity after admission (BDCP type 2)\n3 = primary diagnosis (BDCP type M)\n4 = secondary diagnosis (BDCP type 3)\n\nReference: Canadian Norms for Codification of Medical Diagnoses, https://bit.ly/346yCHg';

COMMENT ON COLUMN covidb.diagnosis_data.diagnosis_name IS 'Name of diagnosis that was recorded.\n\nCould be in: { viral pneumonitis, bacterial pneumonia, acute respiratory distress syndrome, pneumothorax, pleural effusion, cryptogenic organizing pneumonia, bronchiolitis, meningitis, seizure, stroke, congestive heart failure, endocarditis, myocarditis, pericarditis, atrial fibrillation, ventricular tachycardia/ventricular fibrillation, STEMI, NSTEMI, cardiac arrest, bactermia, disseminated intravascular coagulation, anemia, rhabdomyolysis, acute renal injury, acute renal failure, gastrointestinal hemorrhage, pancreatitis, liver dysfunction, hyperglycemia, hypoglycemia, home oxygen therapy, outpatient dialysis }';

COMMENT ON COLUMN covidb.diagnosis_data.diagnosis_icd_code IS 'ICD-10 code referencing the diagnosis.';

COMMENT ON COLUMN covidb.diagnosis_data.diagnosis_source IS 'Data source from which the diagnosis occurred.';

ALTER TABLE covidb.culture_data ADD CONSTRAINT fk_culture_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.diagnosis_data ADD CONSTRAINT fk_diagnosis_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.diagnosis_data ADD CONSTRAINT fk_diagnosis_data_episode_data FOREIGN KEY ( episode_data_id ) REFERENCES covidb.episode_data( episode_data_id );

ALTER TABLE covidb.drug_data ADD CONSTRAINT fk_drug_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.episode_data ADD CONSTRAINT fk_episode_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.exposure_data ADD CONSTRAINT fk_exposure_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.exposure_data ADD CONSTRAINT fk_exposure_data_exposure_type FOREIGN KEY ( exposure_type_id ) REFERENCES covidb.exposure_type( exposure_type_id );

ALTER TABLE covidb.imaging_data ADD CONSTRAINT fk_imaging_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.intervention_data ADD CONSTRAINT fk_intervention_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.intervention_data ADD CONSTRAINT fk_intervention_data_drug_data FOREIGN KEY ( drug_data_id ) REFERENCES covidb.drug_data( drug_data_id );

ALTER TABLE covidb.intervention_type ADD CONSTRAINT fk_intervention_data_intervention_type FOREIGN KEY ( intervention_type_id ) REFERENCES covidb.intervention_data( intervention_type_id );

ALTER TABLE covidb.lab_data ADD CONSTRAINT fk_lab_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.observation_data ADD CONSTRAINT fk_observation_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.observation_data ADD CONSTRAINT fk_observation_data_observation_type FOREIGN KEY ( observation_type_id ) REFERENCES covidb.observation_type( observation_type_id );

ALTER TABLE covidb.pcr_data ADD CONSTRAINT fk_pcr_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.signal_data ADD CONSTRAINT fk_signal_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.slice_data ADD CONSTRAINT fk_slice_data_imaging_data FOREIGN KEY ( imaging_data_id ) REFERENCES covidb.imaging_data( imaging_data_id );

ALTER TABLE covidb.slice_data ADD CONSTRAINT fk_slice_data_culture_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );

ALTER TABLE covidb.symptom_data ADD CONSTRAINT fk_symptom_data_patient_data FOREIGN KEY ( patient_uid ) REFERENCES covidb.patient_data( patient_uid );
