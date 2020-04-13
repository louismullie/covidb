CREATE TABLE exposure_type ( 
	exposure_type_id     bigint NOT NULL  PRIMARY KEY  ,
	exposure_type_name   varchar(100)     ,
	exposure_type_description varchar     ,
	exposure_type_total_units varchar     
 );

CREATE TABLE observation_type ( 
	observation_type_id  bigint NOT NULL  PRIMARY KEY  ,
	observation_category smallint     ,
	observation_name     varchar NOT NULL    ,
	observation_units    varchar NOT NULL    ,
	observation_regexp   varchar     
 );

CREATE TABLE patient_data ( 
	patient_site_uid     char(128) NOT NULL  PRIMARY KEY  ,
	patient_global_uid   char(128) NOT NULL    ,
	patient_site_code    varchar     ,
	patient_transfer_site_code date     ,
	patient_covid_status smallint     ,
	patient_birth_date   date     ,
	patient_sex          char(1)     ,
	patient_birth_country char(2)     ,
	patient_ethnicity    smallint     ,
	patient_pregnant     boolean     ,
	patient_postal_district char(3)     ,
	patient_assisted_living smallint     ,
	patient_code_status  smallint     ,
	patient_death        boolean     ,
	patient_death_cause  varchar     ,
	patient_last_fu_date date     ,
	CONSTRAINT Unq_patient_data_patient_site_uid UNIQUE ( patient_site_uid ) 
 );

CREATE TABLE pcr_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	pcr_data_id          bigint NOT NULL  PRIMARY KEY  ,
	pcr_name             varchar(100)     ,
	pcr_status           smallint     ,
	pcr_loinc_code       varchar     ,
	pcr_sample_time      varchar     ,
	pcr_sample_site      smallint     ,
	pcr_result_time      varchar     ,
	pcr_result_positive  boolean     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE signal_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	signal_data_id       bigint NOT NULL  PRIMARY KEY  ,
	signal_type          smallint     ,
	signal_num_channels  smallint     ,
	signal_sampling_freq smallint     ,
	signal_data_uri      varchar     ,
	signal_start_time    varchar     ,
	signal_end_time      varchar     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE symptom_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	symptom_data_id      bigint NOT NULL  PRIMARY KEY  ,
	symptom_snomed_code  varchar     ,
	symptom_name         varchar(100)     ,
	symptom_details      varchar     ,
	symptom_source       varchar     ,
	symptom_start_time   varchar     ,
	symptom_end_time     varchar     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE culture_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	culture_data_id      integer NOT NULL  PRIMARY KEY  ,
	culture_type         smallint     ,
	culture_status       smallint     ,
	culture_loinc_code   varchar     ,
	culture_sample_site  smallint     ,
	culture_sample_time  varchar     ,
	culture_result_time  varchar     ,
	culture_gram_positive boolean     ,
	culture_growth_positive boolean     ,
	culture_result_organisms boolean     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE drug_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	drug_data_id         bigint NOT NULL  PRIMARY KEY  ,
	drug_name            varchar(100)     ,
	drug_din_code        varchar     ,
	drug_class           varchar     ,
	drug_infusion        boolean     ,
	drug_dose            varchar     ,
	drug_rate            float     ,
	drug_roa             varchar     ,
	drug_frequency       smallint     ,
	drug_start_time      varchar     ,
	drug_end_time        varchar     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE episode_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	episode_data_id      bigint NOT NULL  PRIMARY KEY  ,
	episode_unit_type    smallint     ,
	episode_ctas         date     ,
	episode_start_time   varchar     ,
	episode_end_time     varchar     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE exposure_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	exposure_data_id     integer NOT NULL  PRIMARY KEY  ,
	exposure_type_id     integer     ,
	exposure_total       varchar     ,
	exposure_details     varchar     ,
	exposure_start_time  varchar     ,
	exposure_end_time    varchar     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  ,
	FOREIGN KEY ( exposure_type_id ) REFERENCES exposure_type( exposure_type_id )  
 );

CREATE TABLE imaging_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	imaging_data_id      bigint NOT NULL  PRIMARY KEY  ,
	imaging_accession_number varchar     ,
	imaging_modality     varchar     ,
	imaging_iv_contrast  boolean     ,
	imaging_manufacturer varchar     ,
	imaging_acquisition_time varchar     ,
	imaging_report_time  varchar     ,
	imaging_report_text  varchar(250)     ,
	imaging_site         varchar(36)     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE intervention_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	drug_data_id         integer     ,
	intervention_data_id bigint NOT NULL  PRIMARY KEY  ,
	intervention_type_id smallint     ,
	intervention_details varchar(100)     ,
	intervention_start_time varchar     ,
	intervention_end_time varchar     ,
	CONSTRAINT unq_intervention_data_intervention_type_id UNIQUE ( intervention_type_id ) ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  ,
	FOREIGN KEY ( drug_data_id ) REFERENCES drug_data( drug_data_id )  
 );

CREATE TABLE intervention_type ( 
	intervention_type_id bigint NOT NULL  PRIMARY KEY  ,
	intervention_name    varchar     ,
	intervention_invasiveness bigint     ,
	FOREIGN KEY ( intervention_type_id ) REFERENCES intervention_data( intervention_type_id )  
 );

CREATE TABLE lab_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	lab_data_id          bigint NOT NULL  PRIMARY KEY  ,
	lab_name             varchar(100)     ,
	lab_loinc_code       varchar     ,
	lab_cerner_code      varchar     ,
	lab_units            varchar     ,
	lab_status           smallint     ,
	lab_sample_time      varchar     ,
	lab_sample_type      smallint     ,
	lab_result_time      varchar     ,
	lab_result_value     float     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  
 );

CREATE TABLE observation_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	observation_data_id  bigint NOT NULL  PRIMARY KEY  ,
	observation_type_id  integer NOT NULL    ,
	observation_device   varchar     ,
	observation_start_time varchar     ,
	observation_end_time varchar     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  ,
	FOREIGN KEY ( observation_type_id ) REFERENCES observation_type( observation_type_id )  
 );

CREATE TABLE slice_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	imaging_data_id      integer     ,
	slice_data_id        bigint NOT NULL  PRIMARY KEY  ,
	slice_series_id      varchar     ,
	slice_data_uri       varchar     ,
	slice_view_position  varchar     ,
	slice_patient_position varchar     ,
	slice_image_orientation varchar     ,
	slice_image_position varchar     ,
	slice_window_center  varchar     ,
	slice_window_width   varchar     ,
	slice_pixel_spacing  varchar     ,
	slice_thickness      varchar     ,
	FOREIGN KEY ( imaging_data_id ) REFERENCES imaging_data( imaging_data_id )  
 );

CREATE TABLE diagnosis_data ( 
	patient_site_uid     char(128) NOT NULL    ,
	episode_data_id      integer     ,
	diagnosis_id         bigint NOT NULL  PRIMARY KEY  ,
	diagnosis_type       varchar     ,
	diagnosis_name       varchar(100)     ,
	diagnosis_icd_code   varchar     ,
	diagnosis_source     varchar     ,
	FOREIGN KEY ( patient_site_uid ) REFERENCES patient_data( patient_site_uid )  ,
	FOREIGN KEY ( episode_data_id ) REFERENCES episode_data( episode_data_id )  
 );

