CREATE TABLE exposure_type ( 
	exposure_type_id     integer NOT NULL  PRIMARY KEY  ,
	exposure_type_name   text     ,
	exposure_type_description text     ,
	exposure_type_cumulative_units text     
 );

CREATE TABLE observation_type ( 
	observation_type_id  integer NOT NULL  PRIMARY KEY  ,
	observation_type_class smallint     ,
	observation_type_name text NOT NULL    ,
	observation_units    text NOT NULL    ,
	observation_regexp   text     
 );

CREATE TABLE patient_data ( 
	patient_site_uid     char(128) NOT NULL  PRIMARY KEY  ,
	patient_global_uid   char(128) NOT NULL    ,
	patient_site_code    text     ,
	patient_was_transferred boolean     ,
	patient_transfer_site_code text     ,
	patient_covid_status text     ,
	patient_birth_date   date     ,
	patient_sex          text     ,
	patient_birth_country char(2)     ,
	patient_ethnicity    smallint     ,
	patient_pregnant     boolean     ,
	patient_postal_district char(3)     ,
	patient_assisted_living boolean     ,
	patient_code_status  smallint     ,
	patient_vital_status text     ,
	patient_last_fu_date date     ,
	CONSTRAINT Unq_patient_data_patient_site_uid UNIQUE ( patient_site_uid ) 
 );

CREATE TABLE pcr_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	pcr_data_id          integer NOT NULL  PRIMARY KEY  ,
	pcr_name             text     ,
	pcr_loinc_code       text     ,
	pcr_sample_time      timestamp     ,
	pcr_sample_type      text     ,
	pcr_result_time      timestamp     ,
	pcr_result_status    text     ,
	pcr_result_value     text     ,
	pcr_result_transferred boolean     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE signal_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	signal_data_id       integer NOT NULL  PRIMARY KEY  ,
	signal_name          text     ,
	signal_num_channels  smallint     ,
	signal_sampling_freq float     ,
	signal_data_uri      text     ,
	signal_start_time    timestamp     ,
	signal_end_time      text     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE symptom_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	symptom_data_id      integer NOT NULL  PRIMARY KEY  ,
	symptom_snomed_code  text     ,
	symptom_name         text     ,
	symptom_details      text     ,
	symptom_source       text     ,
	symptom_start_time   timestamp     ,
	symptom_end_time     timestamp     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE culture_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	culture_data_id      integer NOT NULL  PRIMARY KEY  ,
	culture_type         text     ,
	culture_result_status text     ,
	culture_loinc_code   text     ,
	culture_sample_type  text     ,
	culture_sample_time  timestamp     ,
	culture_result_time  timestamp     ,
	culture_gram_positive text     ,
	culture_growth_result text     ,
	culture_result_organisms boolean     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE drug_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	drug_data_id         integer NOT NULL  PRIMARY KEY  ,
	drug_name            text     ,
	drug_din_code        text     ,
	drug_class           text     ,
	drug_infusion        boolean     ,
	drug_dose            text     ,
	drug_rate            text     ,
	drug_roa             text     ,
	drug_frequency       text     ,
	drug_start_time      timestamp     ,
	drug_end_time        timestamp     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE episode_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	episode_data_id      integer NOT NULL  PRIMARY KEY  ,
	episode_unit_type    text     ,
	episode_ctas         smallint     ,
	episode_start_time   timestamp     ,
	episode_end_time     timestamp     ,
	episode_description  text     ,
	episode_duration_hours integer     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE exposure_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	exposure_data_id     integer NOT NULL  PRIMARY KEY  ,
	exposure_type_id     integer     ,
	exposure_cumulative  float     ,
	exposure_details     text     ,
	exposure_start_time  timestamp     ,
	exposure_end_time    timestamp     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  ,
	FOREIGN KEY ( exposure_type_id ) REFERENCES exposure_type( exposure_type_id )  
 );

CREATE TABLE imaging_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	imaging_data_id      integer NOT NULL  PRIMARY KEY  ,
	imaging_accession_uid char(128)     ,
	imaging_status       text     ,
	imaging_modality     text     ,
	imaging_body_site    text     ,
	imaging_iv_contrast  boolean     ,
	imaging_manufacturer text     ,
	imaging_acquisition_time timestamp     ,
	imaging_report_time  timestamp     ,
	imaging_report_text  text     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE intervention_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	drug_data_id         integer     ,
	intervention_data_id integer NOT NULL  PRIMARY KEY  ,
	intervention_type_id integer     ,
	intervention_details text     ,
	intervention_start_time time     ,
	intervention_end_time timestamp     ,
	CONSTRAINT unq_intervention_data_intervention_type_id UNIQUE ( intervention_type_id ) ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  ,
	FOREIGN KEY ( drug_data_id ) REFERENCES drug_data( drug_data_id )  
 );

CREATE TABLE intervention_type ( 
	intervention_type_id integer NOT NULL  PRIMARY KEY  ,
	intervention_name    text     ,
	FOREIGN KEY ( intervention_type_id ) REFERENCES intervention_data( intervention_type_id )  
 );

CREATE TABLE lab_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	lab_data_id          integer NOT NULL  PRIMARY KEY  ,
	lab_name             text     ,
	lab_loinc_code       text     ,
	lab_cerner_code      text     ,
	lab_units            text     ,
	lab_sample_time      timestamp     ,
	lab_sample_type      text     ,
	lab_result_status    text     ,
	lab_result_time      timestamp     ,
	lab_result_value     float     ,
	lab_result_value_string text     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE observation_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	observation_data_id  integer NOT NULL  PRIMARY KEY  ,
	observation_type_id  integer NOT NULL    ,
	observation_device   text     ,
	observation_source   text     ,
	observation_start_time time     ,
	observation_end_time timestamp     ,
	observation_value    float     ,
	observation_value_string text     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  ,
	FOREIGN KEY ( observation_type_id ) REFERENCES observation_type( observation_type_id )  
 );

CREATE TABLE slice_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	imaging_data_id      integer     ,
	imaging_accession_uid char(128)     ,
	slice_data_id        bigint NOT NULL  PRIMARY KEY  ,
	slice_series_instance_uid char(128)     ,
	slice_study_instance_uid char(128)     ,
	slice_data_uri       text     ,
	slice_view_position  text     ,
	slice_patient_position text     ,
	slice_image_orientation text     ,
	slice_image_position text     ,
	slice_window_center  float     ,
	slice_window_width   float     ,
	slice_pixel_spacing  text     ,
	slice_thickness      float     ,
	slice_rows           integer     ,
	slice_columns        integer     ,
	slice_rescale_intercept float     ,
	slice_rescale_slope  float     ,
	FOREIGN KEY ( imaging_data_id ) REFERENCES imaging_data( imaging_data_id )  
 );

CREATE TABLE diagnosis_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	episode_data_id      integer     ,
	diagnosis_data_id    integer NOT NULL  PRIMARY KEY  ,
	diagnosis_type       text     ,
	diagnosis_name       text     ,
	diagnosis_icd_code   text     ,
	diagnosis_source     text     ,
	diagnosis_time       timestamp     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  ,
	FOREIGN KEY ( episode_data_id ) REFERENCES episode_data( episode_data_id )  
 );

