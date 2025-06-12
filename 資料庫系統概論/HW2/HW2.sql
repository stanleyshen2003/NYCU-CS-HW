4a:
(select maxindex.ct as continent,alld.date, alld.country_name
from (select max(stringency.stringencyindex) as maxstr,country_continent.continent as ct
		from (select *
		from oxcgrt_data
		where oxcgrt_data.date='20200601')as ad natural join stringency natural join country_continent
		group by continent) as maxindex,
		((select * from oxcgrt_data
		where oxcgrt_data.date='20200601')as ad natural join stringency natural join country_continent)
		as alld
where alld.stringencyindex=maxindex.maxstr and alld.continent=maxindex.ct)
union
(select maxindex.ct as continent,alld.date, alld.country_name
from (select max(stringency.stringencyindex) as maxstr,country_continent.continent as ct
		from (select *
		from oxcgrt_data
		where oxcgrt_data.date='20210601')as ad natural join stringency natural join country_continent
		group by continent) as maxindex,
		((select * from oxcgrt_data
		where oxcgrt_data.date='20210601')as ad natural join stringency natural join country_continent)
		as alld
where alld.stringencyindex=maxindex.maxstr and alld.continent=maxindex.ct)
union
(select maxindex.ct as continent,alld.date, alld.country_name
from (select max(stringency.stringencyindex) as maxstr,country_continent.continent as ct
		from (select *
		from oxcgrt_data
		where oxcgrt_data.date='20220601')as ad natural join stringency natural join country_continent
		group by continent) as maxindex,
		((select * from oxcgrt_data
		where oxcgrt_data.date='20220601')as ad natural join stringency natural join country_continent)
		as alld
where alld.stringencyindex=maxindex.maxstr and alld.continent=maxindex.ct)
order by date,continent

-------------------------------------------------------------------------------------------------------------

4b:
(select showT.country_name,showT.continent,'20200601' as date
from (select coun_conti_moving.continent as con,min(coun_conti_moving.overindex) as minindex
	from(select b.stringencyindex/tocon.totalconfirm as overindex,tocon.country_name, country_continent.continent
		from(select sum(confirmedcases) as totalconfirm, country_name
 			from oxcgrt_data	
			where oxcgrt_data.date='20200601' or oxcgrt_data.date='20200531' or oxcgrt_data.date='20200530' or oxcgrt_data.date='20200529' or oxcgrt_data.date='20200528' or oxcgrt_data.date='20200527' or oxcgrt_data.date='20200526'
			group by country_name) as tocon ,
			(select stringencyindex,country_name 
			 from (select * 
				   from oxcgrt_data 
				   where oxcgrt_data.date='20200601') as newt natural join stringency) as b,country_continent
		where tocon.country_name = b.country_name and country_continent.country_name=tocon.country_name) as coun_conti_moving
	group by coun_conti_moving.continent) as referenceT,
	(select b.stringencyindex/tocon.totalconfirm as overindex,tocon.country_name, country_continent.continent
		from(select sum(confirmedcases) as totalconfirm, country_name
 			from oxcgrt_data	
			where oxcgrt_data.date='20200601' or oxcgrt_data.date='20200531' or oxcgrt_data.date='20200530' or oxcgrt_data.date='20200529' or oxcgrt_data.date='20200528' or oxcgrt_data.date='20200527' or oxcgrt_data.date='20200526'
			group by country_name) as tocon ,
			(select stringencyindex,country_name 
			 from (select * 
				   from oxcgrt_data 
				   where oxcgrt_data.date='20200601') as newt natural join stringency) as b,country_continent
		where tocon.country_name = b.country_name and country_continent.country_name=tocon.country_name) as showT
where showT.continent=referenceT.con and showT.overindex = referenceT.minindex)
UNION
(select showT.country_name,showT.continent,'20210601' as date
from (select coun_conti_moving.continent as con,min(coun_conti_moving.overindex) as minindex
	from(select b.stringencyindex/tocon.totalconfirm as overindex,tocon.country_name, country_continent.continent
		from(select sum(confirmedcases) as totalconfirm, country_name
 			from oxcgrt_data	
			where oxcgrt_data.date='20210601' or oxcgrt_data.date='20210531' or oxcgrt_data.date='20210530' or oxcgrt_data.date='20210529' or oxcgrt_data.date='20210528' or oxcgrt_data.date='20210527' or oxcgrt_data.date='20210526'
			group by country_name) as tocon ,
			(select stringencyindex,country_name 
			 from (select * 
				   from oxcgrt_data 
				   where oxcgrt_data.date='20210601') as newt natural join stringency) as b,country_continent
		where tocon.country_name = b.country_name and country_continent.country_name=tocon.country_name) as coun_conti_moving
	group by coun_conti_moving.continent) as referenceT,
	(select b.stringencyindex/tocon.totalconfirm as overindex,tocon.country_name, country_continent.continent
		from(select sum(confirmedcases) as totalconfirm, country_name
 			from oxcgrt_data	
			where oxcgrt_data.date='20210601' or oxcgrt_data.date='20210531' or oxcgrt_data.date='20210530' or oxcgrt_data.date='20210529' or oxcgrt_data.date='20210528' or oxcgrt_data.date='20210527' or oxcgrt_data.date='20210526'
			group by country_name) as tocon ,
			(select stringencyindex,country_name 
			 from (select * 
				   from oxcgrt_data 
				   where oxcgrt_data.date='20210601') as newt natural join stringency) as b,country_continent
		where tocon.country_name = b.country_name and country_continent.country_name=tocon.country_name) as showT
where showT.continent=referenceT.con and showT.overindex = referenceT.minindex)
union
(select showT.country_name,showT.continent,'20220601' as date
from (select coun_conti_moving.continent as con,min(coun_conti_moving.overindex) as minindex
	from(select b.stringencyindex/tocon.totalconfirm as overindex,tocon.country_name, country_continent.continent
		from(select sum(confirmedcases) as totalconfirm, country_name
 			from oxcgrt_data	
			where oxcgrt_data.date='20220601' or oxcgrt_data.date='20220531' or oxcgrt_data.date='20220530' or oxcgrt_data.date='20220529' or oxcgrt_data.date='20220528' or oxcgrt_data.date='20220527' or oxcgrt_data.date='20220526'
			group by country_name) as tocon ,
			(select stringencyindex,country_name 
			 from (select * 
				   from oxcgrt_data 
				   where oxcgrt_data.date='20220601') as newt natural join stringency) as b,country_continent
		where tocon.country_name = b.country_name and country_continent.country_name=tocon.country_name) as coun_conti_moving
	group by coun_conti_moving.continent) as referenceT,
	(select b.stringencyindex/tocon.totalconfirm as overindex,tocon.country_name, country_continent.continent
		from(select sum(confirmedcases) as totalconfirm, country_name
 			from oxcgrt_data	
			where oxcgrt_data.date='20220601' or oxcgrt_data.date='20220531' or oxcgrt_data.date='20220530' or oxcgrt_data.date='20220529' or oxcgrt_data.date='20220528' or oxcgrt_data.date='20220527' or oxcgrt_data.date='20220526'
			group by country_name) as tocon ,
			(select stringencyindex,country_name 
			 from (select * 
				   from oxcgrt_data 
				   where oxcgrt_data.date='20220601') as newt natural join stringency) as b,country_continent
		where tocon.country_name = b.country_name and country_continent.country_name=tocon.country_name) as showT
where showT.continent=referenceT.con and showT.overindex = referenceT.minindex)
order by date