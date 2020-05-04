
db_file_name = os.path.join(SQLITE_DIRECTORY, 'covidb_version-1.0.0.db')
conn = sqlite3.connect(db_file_name)

query = "SELECT patient_site_uid from patient_data"

mrns = sql_fetch_all(conn, query)
ms = list(set(([str(r[0]) for r in mrns])))

print('Total number of patients tested: %d' % len(ms))

query = "SELECT patient_site_uid from patient_data WHERE " + \
         " patient_data.patient_covid_status = 'positive'"

mrns = sql_fetch_all(conn, query)
ms = list(set(([str(r[0]) for r in mrns])))

query = "SELECT episode_data.patient_site_uid from episode_data INNER JOIN " + \
         " patient_data ON episode_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
         " (episode_data.episode_unit_type = 'inpatient_ward' OR " + \
         "  episode_data.episode_unit_type = 'coronary_care_unit' OR " + \
         "  episode_data.episode_unit_type = 'high_dependency_unit') AND " + \
         " patient_data.patient_covid_status = 'positive'"

mrns = sql_fetch_all(conn, query)
ms = set(([str(r[0]) for r in mrns]))
num_nonicu = len(ms)

query = "SELECT episode_data.patient_site_uid from episode_data INNER JOIN " + \
         " patient_data ON episode_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
         " episode_data.episode_unit_type = 'intensive_care_unit' AND " + \
         " patient_data.patient_covid_status = 'positive'"

mrns = sql_fetch_all(conn, query)
ms = set(([str(r[0]) for r in mrns]))
print('Number of ICU hospitalized patients: %d' % len(ms))
num_icu = len(ms)

query = "SELECT episode_data.patient_site_uid from episode_data INNER JOIN " + \
         " patient_data ON episode_data.patient_site_uid = patient_data.patient_site_uid WHERE " + \
         " episode_data.episode_unit_type = 'high_dependency_unit' AND " + \
         " patient_data.patient_covid_status = 'positive'"

mrns = sql_fetch_all(conn, query)
ms = list(set(([str(r[0]) for r in mrns])))
print('Number of hospitalized patients: %d' % (num_icu + num_nonicu))
