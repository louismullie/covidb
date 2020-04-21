def sql_fetch_all(conn, query):

  curr = conn.execute(query)
  res = curr.fetchall()
  return res

def sql_fetch_one(conn, query):

  curr = conn.execute(query)
  res = curr.fetchone()
  return res