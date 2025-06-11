from unittest import TestCase

from string_trans.string_env import StringTransEnv
from string_trans.dsl import StringTransformationDSL

dsl = StringTransformationDSL()
env = StringTransEnv(input_str="")

def exec(program, input_str):
    env.set_new_state(input_str)
    dsl.run(env, program)
    return env.output_str

class TestDSL(TestCase):

    def test_SubStr(self):
        # SubStr merges two functions, `SubStr` and `GetSpan` in RobustFill.
        # The SubStr function takes two positions, and Regex can get the position.
        program = "Concat c( SubStr s( ConstPos p( 0 p) ConstPos p( 5 p) s) c)"
        input_str = "0123456789"
        output_str = "01234"
        self.assertEqual(output_str, exec(program, input_str))

        program = "Concat c( SubStr s( ConstPos p( -5 p) ConstPos p( 6 p) s) c)"
        input_str = "0123456789"
        output_str = "5"
        self.assertEqual(output_str, exec(program, input_str))

        program = "Concat c( SubStr s( Regex r( \"[\" 3 End r) Regex r( \",\" 1 Start r) s) c)"
        input_str = "[[asdf[[qq,002,111234"
        output_str = "qq,002"
        self.assertEqual(output_str, exec(program, input_str))

        program = "Concat c( SubStr s( Regex r( Propcase 0 Start r) Regex r( \"@\" 0 End r) s) c)"
        input_str = "AsdsFFasd134@@@asdf"
        output_str = "AsdsFFasd134@"
        self.assertEqual(output_str, exec(program, input_str))

        program = "Concat c( SubStr s( Regex r( Lower 1 End r) Regex r( Digit -2 End r) s) c)"
        input_str = "AsdsFFasd134@@@asdf"
        output_str = "13"
        self.assertEqual(output_str, exec(program, input_str))

    def test_GetToken(self):
        program = "Concat c( GetToken n( Lower 5 n) c)"
        input_str = "For evaluating the trained models, we use FlashFillTest."
        output_str = "we"
        self.assertEqual(output_str, exec(program, input_str))

        program = "Concat c( GetToken n( Allcaps 3 n) c)"
        input_str = "For evaluating the trained models, we use FlashFillTest."
        output_str = "T"
        self.assertEqual(output_str, exec(program, input_str))

        program = "Concat c( GetToken n( Word 3 n) GetToken n( Alphanum -3 n) c)"
        input_str = "In all experiments, the size of the recurrent and fully connected layers is 512, and the size of the embeddings is 128."
        output_str = "theembeddings"
        self.assertEqual(output_str, exec(program, input_str))

    def test_ToCase(self):
        program = "Concat c( ToCase n( Allcaps n) c)"
        input_str = "For evaluating the trained models, we use FlashFillTest."
        output_str = "FOR EVALUATING THE TRAINED MODELS, WE USE FLASHFILLTEST."
        self.assertEqual(output_str, exec(program, input_str))

    def test_Replace(self):
        program = "Concat c( Replace n( \"SPACE\" \"'\" n) c)"
        input_str = "For evaluating the trained models, we use FlashFillTest."
        output_str = "For'evaluating'the'trained'models,'we'use'FlashFillTest."
        self.assertEqual(output_str, exec(program, input_str))


    def test_Trim(self):
        program = "Concat c( Trim n( n) c)"
        input_str = "\tdef test_Trim(self):"
        output_str = "def test_Trim(self):"
        self.assertEqual(output_str, exec(program, input_str))

    def test_GetUpTo(self):
        program = "Concat c( GetUpTo n( Allcaps n) c)"
        input_str = "\tdef test_GetUpTo(self):"
        output_str = "\tdef test_G"
        self.assertEqual(output_str, exec(program, input_str))

    def test_GetFrom(self):
        program = "Concat c( GetFrom n( \"SPACE\" n) c)"
        input_str = "\tdef test_GetFrom(self):"
        output_str = "test_GetFrom(self):"
        self.assertEqual(output_str, exec(program, input_str))

    def test_GetFirst(self):
        program = "Concat c( GetFirst n( Lower 3 n) c)"
        input_str = "For evaluating the trained models, we use FlashFillTest."
        output_str = "orevaluatingthetrained"
        self.assertEqual(output_str, exec(program, input_str))

    def test_GetAll(self):
        program = "Concat c( GetAll n( Propcase n) c)"
        input_str = "For evaluating the trained models, we use FlashFillTest."
        output_str = "For Flash Fill Test"
        self.assertEqual(output_str, exec(program, input_str))

    def test_ConstStr(self):
        program = "Concat c( ConstStr k( \"@\" k) c)"
        input_str = "asdfasdf"
        output_str = "@"
        self.assertEqual(output_str, exec(program, input_str))

    def test_Nesting(self):
        program = "Concat c( GetAll n( Lower n) v( SubStr s( Regex r( Char 3 Start r) Regex r( \",\" 0 Start r) s) v) c)"
        input_str = "For evaluating the trained models, we use FlashFillTest."
        output_str = "evaluating the trained models"
        self.assertEqual(output_str, exec(program, input_str))

        program = "Concat c( ToCase n( Lower n) v( SubStr s( ConstPos p( 0 p) ConstPos p( 3 p) s) v) c)"
        inputs = ["January", "February", "March", "April"]
        outputs = ["jan", "feb", "mar", "apr"]
        for i, o in zip(inputs, outputs):
            self.assertEqual(o, exec(program, i))

    def test_RobustFillSample0(self):
        # this is the example of Figure 1 with small modify
        inputs = ["john Smith", "DOUG Q. Macklin", "Frank Lee (123)", "Laura Jane Jones"]
        outputs = ["Smith, John", "Macklin, Doug", "Lee, Frank", "Jones, Laura"]
        program = "Concat c( GetToken n( Word -1 n) ConstStr k( \",\" k) ConstStr k( \"SPACE\" k) ToCase n( Proper n) v( GetToken n( Alphanum 0 n) v) c)"
        for i, o in zip(inputs, outputs):
            self.assertEqual(o, exec(program, i))

    def test_RobustFillSample1(self):
        # this is the first examle of Figure 12
        # GetToken_Alphanum_3 | GetFrom_Colon | GetFirst_Char_4
        inputs = ["Ud 9:25,JV3 Obb", "zLny xmHg 8:43 A44q", "A6 g45P 10:63 Jf", "cuL.zF.dDX,12:31", "ZiG OE bj3u 7:11"]
        outputs = ["2525,JV3 ObbUd92", "843 A44qzLny", "1063 JfA6g4", "dDX31cuLz", "bj3u11ZiGO"]
        program = "Concat c( GetToken n( Alphanum 2 n) GetFrom n( \":\" n) GetFirst n( Char 3 n) c)"
        
        for i, o in zip(inputs, outputs):
            self.assertEqual(o, exec(program, i))

    def test_RobustFillSample2(self):
        # this is the third examle of Figure 12
        # GetToken_AllCaps_-2(GetSpan(AllCaps, 1, Start, AllCaps, 5, Start))
        inputs = ["YDXJZ @ZYUD Wc-YKT GTIL BNX",
                  "JUGRB.MPKA.MTHV,tEczT-GZJ.MFT",
                  "VXO.OMQDK.JC-OAR,HZGH-DJKC",
                  "HCUD-WDOC,RTTRQ-KVETK-whx-DIKDI",
                  "JFNB.Avj,ODZBT-XHV,KYB @,RHVVW"
                  ]
        outputs = ["W", "MTHV", "JC", "RTTRQ", "ODZBT"]
        program = "Concat c( GetToken n( Allcaps -2 n) v( SubStr s( Regex r( Allcaps 0 Start r) Regex r( Allcaps 4 Start r) s) v) c)"
        for i, o in zip(inputs, outputs):
            self.assertEqual(o, exec(program, i))

    def test_RobustFillSample3(self):
        # this is the forth examle of Figure 12
        #  SubStr(-20, -8) | GetToken_AllCaps_-3 | SubStr(11, 19) | GetToken_Alphanum_-5
        inputs = ["DvD 6X xkd6 OZQIN ZZUK,nCF aQR IOHR",
                  "BHP-euSZ,yy,44-CRCUC,ONFZA.mgOJ.Hwm",
                  "NGM-8nay,xrL.GmOc.PFLH,CMFEX-JPFA,iIcj,329",
                  "hU TQFLD Lycb NCPYJ oo FS TUM l6F",
                  "OHHS NNDQ XKQRN KDL 8Ucj dUqh Cpk Kafj"
                  ]
        outputs = ["IN ZZUK,nCF aCF6 OZQIN ZOZQIN",
                   "CRCUC,ONFZA.mONFZAy,44-CRCU44",
                   ",CMFEX-JPFA,iCMFEXrL.GmOc.PPFLH",
                   " NCPYJ oo FS FSycb NCPYJNCPYJ",
                   "L 8Ucj dUqh CUXKQRN KDLKDL"
                   ]
        program = "Concat c( SubStr s( ConstPos p( -20 p) ConstPos p( -7 p) s) GetToken n( Allcaps -3 n) SubStr s( ConstPos p( 10 p) ConstPos p( 19 p) s) GetToken n( Alphanum -5 n) c)"
        for i, o in zip(inputs, outputs):
            self.assertEqual(o, exec(program, i))
