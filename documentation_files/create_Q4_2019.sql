--
-- Create a table where the column names exactly match
-- the column names in the published CSV files.  This is for Q4 of 2019.
--
CREATE TABLE drive_stats (
    date TEXT NOT NULL,
    serial_number TEXT NOT NULL,
    model TEXT NOT NULL,
    capacity_bytes INTEGER (8) NOT NULL,
    failure INTEGER (1) NOT NULL,
    smart_1_normalized INTEGER,
    smart_1_raw INTEGER,
    smart_2_normalized INTEGER,
    smart_2_raw INTEGER,
    smart_3_normalized INTEGER,
    smart_3_raw INTEGER,
    smart_4_normalized INTEGER,
    smart_4_raw INTEGER,
    smart_5_normalized INTEGER,
    smart_5_raw INTEGER,
    smart_7_normalized INTEGER,
    smart_7_raw INTEGER,
    smart_8_normalized INTEGER,
    smart_8_raw INTEGER,
    smart_9_normalized INTEGER,
    smart_9_raw INTEGER,
    smart_10_normalized INTEGER,
    smart_10_raw INTEGER,
    smart_11_normalized INTEGER,
    smart_11_raw INTEGER,
    smart_12_normalized INTEGER,
    smart_12_raw INTEGER,
    smart_13_normalized INTEGER,
    smart_13_raw INTEGER,
    smart_15_normalized INTEGER,
    smart_15_raw INTEGER,
    smart_16_normalized INTEGER,
    smart_16_raw INTEGER,    
    smart_17_normalized INTEGER,
    smart_17_raw INTEGER,
    smart_18_normalized INTEGER,
    smart_18_raw INTEGER,
    smart_22_normalized INTEGER,
    smart_22_raw INTEGER,
    smart_23_normalized INTEGER,
    smart_23_raw INTEGER,
	smart_24_normalized INTEGER,
    smart_24_raw INTEGER,
    smart_168_normalized INTEGER,
    smart_168_raw INTEGER,        
    smart_170_normalized INTEGER,
    smart_170_raw INTEGER,        
    smart_173_normalized INTEGER,
    smart_173_raw INTEGER,       
    smart_174_normalized INTEGER,
    smart_174_raw INTEGER,            
    smart_177_normalized INTEGER,
    smart_177_raw INTEGER,    
    smart_179_normalized INTEGER,
    smart_179_raw INTEGER,    
    smart_181_normalized INTEGER,
    smart_181_raw INTEGER,    
    smart_182_normalized INTEGER,
    smart_182_raw INTEGER,    
    smart_183_normalized INTEGER,
    smart_183_raw INTEGER,
    smart_184_normalized INTEGER,
    smart_184_raw INTEGER,
    smart_187_normalized INTEGER,
    smart_187_raw INTEGER,
    smart_188_normalized INTEGER,
    smart_188_raw INTEGER,
    smart_189_normalized INTEGER,
    smart_189_raw INTEGER,
    smart_190_normalized INTEGER,
    smart_190_raw INTEGER,
    smart_191_normalized INTEGER,
    smart_191_raw INTEGER,
    smart_192_normalized INTEGER,
    smart_192_raw INTEGER,
    smart_193_normalized INTEGER,
    smart_193_raw INTEGER,
    smart_194_normalized INTEGER,
    smart_194_raw INTEGER,
    smart_195_normalized INTEGER,
    smart_195_raw INTEGER,
    smart_196_normalized INTEGER,
    smart_196_raw INTEGER,
    smart_197_normalized INTEGER,
    smart_197_raw INTEGER,
    smart_198_normalized INTEGER,
    smart_198_raw INTEGER,
    smart_199_normalized INTEGER,
    smart_199_raw INTEGER,
    smart_200_normalized INTEGER,
    smart_200_raw INTEGER,
    smart_201_normalized INTEGER,
    smart_201_raw INTEGER,
    smart_218_normalized INTEGER,
    smart_218_raw INTEGER,        
    smart_220_normalized INTEGER,
    smart_220_raw INTEGER,
    smart_222_normalized INTEGER,
    smart_222_raw INTEGER,
    smart_223_normalized INTEGER,
    smart_223_raw INTEGER,
    smart_224_normalized INTEGER,
    smart_224_raw INTEGER,   
    smart_225_normalized INTEGER,
    smart_225_raw INTEGER,
    smart_226_normalized INTEGER,
    smart_226_raw INTEGER,
    smart_231_normalized INTEGER,
    smart_231_raw INTEGER,
    smart_232_normalized INTEGER,
    smart_232_raw INTEGER,
    smart_233_normalized INTEGER,
    smart_233_raw INTEGER,    
    smart_235_normalized INTEGER,
    smart_235_raw INTEGER,    
    smart_240_normalized INTEGER,
    smart_240_raw INTEGER,
    smart_241_normalized INTEGER,
    smart_241_raw INTEGER,
    smart_242_normalized INTEGER,
    smart_242_raw INTEGER,
    smart_250_normalized INTEGER,
    smart_250_raw INTEGER,
    smart_251_normalized INTEGER,
    smart_251_raw INTEGER,
    smart_252_normalized INTEGER,
    smart_252_raw INTEGER,
    smart_254_normalized INTEGER,
    smart_254_raw INTEGER,
    smart_255_normalized INTEGER,
    smart_255_raw INTEGER
    );
