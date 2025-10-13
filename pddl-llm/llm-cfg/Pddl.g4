/**
 * Original header:
 *
 *  PDDL grammar for ANTLR v3
 *  Zeyn Saigol
 *  School of Computer Science
 *  University of Birmingham
 *
 *  $Id: Pddl.g 120 2008-10-02 14:59:50Z zas $
 *
 *
 * Revised and adjusted by Dominik Schreiber, 2018-2019
 */
grammar Pddl;

/************* Start of grammar *******************/ 
pddlDoc
   : domain | problem
   ;

/************* DOMAINS ****************************/ 
domain
   : '(' DEFINE domainName requireDef? typesDef? constantsDef? predicatesDef? functionsDef? constraints? structureDef* ')'
   ;

domainName
   : '(' DOMAIN NAME ')'
   ;

requireDef
   : '(' ':' REQUIREMENTS REQUIRE_KEY+ ')'
   ;

typesDef
   : '(' ':' TYPES typedNameList ')'
   ;

// If have any typed names, they must come FIRST!
typedNameList
   : ( NAME* | singleTypeNameList+ NAME* )
   ;

singleTypeNameList
   : ( NAME+ '-' t = type )
   ;

type
   : ( '(' EITHER primType+ ')' ) | primType
   ;

primType
   : NAME
   ;

functionsDef
   : '(' ':' FUNCTIONS functionList ')'
   ;

functionList
   : ( atomicFunctionSkeleton+ ( '-' functionType )? )*
   ;

atomicFunctionSkeleton
   : '(' functionSymbol typedVariableList ')'
   ;

functionSymbol
   : NAME
   ;

functionType
   : STR_NUMBER
   ;

constantsDef
   : '(' ':' CONSTANTS typedNameList ')'
   ;

predicatesDef
   : '(' ':' PREDICATES atomicFormulaSkeleton+ ')'
   ;

atomicFormulaSkeleton
   : '(' predicate typedVariableList ')'
   ;

predicate
   : NAME
   ;

// If have any typed variables, they must come FIRST!
typedVariableList
   : ( VARIABLE* | singleTypeVarList+ VARIABLE* )
   ;

singleTypeVarList
   : ( VARIABLE+ '-' t = type )
   ;

constraints
   : '(' ':' CONSTRAINTS conGD ')'
   ;

structureDef
   : actionDef | durativeActionDef | derivedDef
   ;

/************* ACTIONS ****************************/ 
actionDef
   : '(' ':' ACTION actionSymbol ':' PARAMETERS '(' typedVariableList ')' actionDefBody ')'
   ;

actionSymbol
   : NAME
   ;

// Should allow preGD instead of goalDesc for preconditions -
// but I can't get the LL(*) parsing to work
// This means 'preference' preconditions cannot be used
actionDefBody
   : ( ':' PRECONDITION ( ( '(' ')' ) | goalDesc ) )? ( ':' EFFECT ( ( '(' ')' ) | effect ) )?
   ;

goalDesc
   : atomicTermFormula | '(' AND goalDesc* ')' | '(' OR goalDesc* ')' | '(' NOT goalDesc ')' | '(' IMPLY goalDesc goalDesc ')' | '(' EXISTS '(' typedVariableList ')' goalDesc ')' | '(' FORALL '(' typedVariableList ')' goalDesc ')' | fComp | '(' EQUALS term term ')'
   ;

fComp
   : '(' binaryComp fExp fExp ')'
   ;

atomicTermFormula
   : '(' predicate term* ')'
   ;

term
   : NAME | VARIABLE
   ;

/************* DURATIVE ACTIONS ****************************/ 
durativeActionDef
   : '(' ':' DURATIVE_ACTION actionSymbol ':' PARAMETERS '(' typedVariableList ')' daDefBody ')'
   ;

daDefBody
   : ':' DURATION durationConstraint | ':' CONDITION ( ( '(' ')' ) | daGD ) | ':' EFFECT ( ( '(' ')' ) | daEffect )
   ;

daGD
   : prefTimedGD | '(' AND daGD* ')' | '(' FORALL '(' typedVariableList ')' daGD ')'
   ;

prefTimedGD
   : timedGD | '(' PREFERENCE NAME? timedGD ')'
   ;

timedGD
   : '(' timeSpecifier goalDesc ')' | '(' OVER_ALL goalDesc ')'
   ;

timeSpecifier
   : AT_START | AT_END
   ;

interval
   : OVER_ALL
   ;

/************* DERIVED DEFINITIONS ****************************/ 
derivedDef
   : '(' ':' DERIVED atomicFormulaSkeleton goalDesc ')'
   ;

/************* EXPRESSIONS ****************************/ 
fExp
   : NUMBER | '(' binaryOp fExp fExp ')' | '(' '-' fExp ')' | fHead
   ;

fHead
   : '(' functionSymbol term* ')' | functionSymbol
   ;

effect
   : '(' AND cEffect* ')' | cEffect
   ;

cEffect
   : '(' FORALL '(' typedVariableList ')' effect ')' | '(' WHEN goalDesc condEffect ')' | pEffect
   ;

pEffect
   : '(' assignOp fHead fExp ')' | '(' NOT atomicTermFormula ')' | atomicTermFormula
   ;

condEffect
   : '(' AND pEffect* ')' | pEffect
   ;

// TODO: should these be uppercase & lexer section?
binaryOp
   : '*' | '+' | '-' | '/'
   ;

binaryComp
   : '>' | '<' | '=' | '>=' | '<='
   ;

assignOp
   : ASSIGN | SCALE_UP | SCALE_DOWN | INCREASE | DECREASE
   ;

/************* DURATIONS  ****************************/ durationConstraint
   : '(' AND simpleDurationConstraint+ ')' | '(' ')' | simpleDurationConstraint
   ;

simpleDurationConstraint
   : '(' durOp '?' DURATION durValue ')' | '(' timeSpecifier simpleDurationConstraint ')'
   ;

durOp
   : '<=' | '>=' | '='
   ;

durValue
   : NUMBER | fExp
   ;

daEffect
   : '(' AND daEffect* ')' | timedEffect | '(' FORALL '(' typedVariableList ')' daEffect ')' | '(' WHEN daGD timedEffect ')' | '(' assignOp fHead fExpDA ')'
   ;

timedEffect
   : '(' timeSpecifier daEffect ')' | '(' timeSpecifier fAssignDA ')' | '(' assignOp fHead fExp ')'
   ;

fAssignDA
   : '(' assignOp fHead fExpDA ')'
   ;

fExpDA
   : '(' ( ( binaryOp fExpDA fExpDA ) | ( '-' fExpDA ) ) ')' | '?' DURATION | fExp
   ;

/************* PROBLEMS ****************************/ 
problem
   : '(' DEFINE problemDecl problemDomain requireDef? objectDecl? init goal probConstraints? metricSpec? ')'
   // lengthSpec? This is not defined anywhere in the BNF spec 
   ;

problemDecl
   : '(' PROBLEM NAME ')'
   ;

problemDomain
   : '(' ':' DOMAIN NAME ')'
   ;

objectDecl
   : '(' ':' OBJECTS typedNameList ')'
   ;

init
   : '(' ':' INIT initEl* ')'
   ;

// TODO : should be 'at' NUMBER nameLiteral
// As of now, timed initial literals cannot be used
initEl
   : nameLiteral | '(' '=' fHead NUMBER ')' | '(' NUMBER nameLiteral ')'
   ;

nameLiteral
   : atomicNameFormula | '(' NOT atomicNameFormula ')'
   ;

atomicNameFormula
   : '(' predicate NAME* ')'
   ;

// TODO allow preGD instead of goalDesc
// As of now, 'preference' preconditions cannot be used
//goal : '(' ':goal' preGD ')'  -> ^(GOAL preGD);
goal
   : '(' ':' GOAL goalDesc ')'
   ;

probConstraints
   : '(' ':' CONSTRAINTS prefConGD ')'
   ;

prefConGD
   : '(' AND prefConGD* ')' | '(' FORALL '(' typedVariableList ')' prefConGD ')' | '(' PREFERENCE NAME? conGD ')' | conGD
   ;

metricSpec
   : '(' ':' METRIC optimization metricFExp ')'
   ;

optimization
   : MINIMIZE | MAXIMIZE
   ;

metricFExp
   : '(' binaryOp metricFExp metricFExp ')' | '(' ( '*' | '/' ) metricFExp metricFExp+ ')' | '(' '-' metricFExp ')' | NUMBER | '(' functionSymbol NAME* ')' | functionSymbol | TOTAL_TIME | '(' IS_VIOLATED NAME ')'
   ;

conGD
   : '(' AND conGD* ')' | '(' FORALL '(' typedVariableList ')' conGD ')' | '(' AT_END goalDesc ')' | '(' ALWAYS goalDesc ')' | '(' SOMETIME goalDesc ')' | '(' WITHIN NUMBER goalDesc ')' | '(' AT_MOST_ONCE goalDesc ')' | '(' SOMETIME_AFTER goalDesc goalDesc ')' | '(' SOMETIME_BEFORE goalDesc goalDesc ')' | '(' ALWAYS_WITHIN NUMBER goalDesc goalDesc ')' | '(' HOLD_DURING NUMBER NUMBER goalDesc ')' | '(' HOLD_AFTER NUMBER goalDesc ')'
   ;

/************* LEXER ****************************/ 

REQUIRE_KEY
   : R_STRIPS|R_TYPING|R_NEGATIVE_PRECONDITIONS|R_DISJUNCTIVE_PRECONDITIONS|R_EQUALITY|R_EXISTENTIAL_PRECONDITIONS|R_UNIVERSAL_PRECONDITIONS|R_QUANTIFIED_PRECONDITIONS|R_CONDITIONAL_EFFECTS|R_FLUENTS|R_ADL|R_DURATIVE_ACTIONS|R_DERIVED_PREDICATES|R_TIMED_INITIAL_LITERALS|R_PREFERENCES|R_CONSTRAINTS|R_ACTION_COSTS
   ;   
   
DEFINE: D E F I N E;
DOMAIN: D O M A I N;
PROBLEM: P R O B L E M;
REQUIREMENTS: R E Q U I R E M E N T S;
TYPES: T Y P E S;
EITHER : E I T H E R ;
FUNCTIONS : F U N C T I O N S ;
CONSTANTS : C O N S T A N T S ;
PREDICATES : P R E D I C A T E S ;
CONSTRAINTS : C O N S T R A I N T S ;
ACTION : A C T I O N ;
PARAMETERS : P A R A M E T E R S ;
PRECONDITION : P R E C O N D I T I O N ;
EFFECT : E F F E C T ;
AND : A N D ;
OR : O R ;
NOT : N O T ;
IMPLY : I M P L Y ;
EXISTS : E X I S T S ;
FORALL : F O R A L L ;
DURATIVE_ACTION : D U R A T I V E '-' A C T I O N ;
DURATION : D U R A T I O N ;
CONDITION : C O N D I T I O N ;
PREFERENCE : P R E F E R E N C E ;
OVER_ALL : O V E R A L L ;
AT_START : A T S T A R T ;
AT_END : A T E N D ;
DERIVED : D E R I V E D ;
WHEN : W H E N ;
ASSIGN : A S S I G N ;
INCREASE : I N C R E A S E ;
DECREASE : D E C R E A S E ;
SCALE_UP : S C A L E '-' U P ;
SCALE_DOWN : S C A L E '-' D O W N ;
OBJECTS : O B J E C T S ;
INIT : I N I T ;
GOAL : G O A L ;
METRIC : M E T R I C ;
MINIMIZE : M I N I M I Z E ;
MAXIMIZE : M A X I M I Z E ;
TOTAL_TIME : T O T A L '-' T I M E ;
IS_VIOLATED : I S '-' V I O L A T E D ;
ALWAYS : A L W A Y S ;
SOMETIME : S O M E T I M E ;
WITHIN : W I T H I N ;
AT_MOST_ONCE : A T '-' M O S T '-' O N C E ;
SOMETIME_AFTER : S O M E T I M E '-' A F T E R ;
SOMETIME_BEFORE : S O M E T I M E '-' B E F O R E ;
ALWAYS_WITHIN : A L W A Y S '-' W I T H I N ;
HOLD_DURING : H O L D '-' D U R I N G ;
HOLD_AFTER : H O L D '-' A F T E R  ;
R_STRIPS : ':' S T R I P S ;
R_TYPING : ':' T Y P I N G ;
R_NEGATIVE_PRECONDITIONS : ':' N E G A T I V E '-' P R E C O N D I T I O N S ;
R_DISJUNCTIVE_PRECONDITIONS : ':' D I S J U N C T I V E '-' P R E C O N D I T I O N S ;
R_EQUALITY : ':' E Q U A L I T Y ;
R_EXISTENTIAL_PRECONDITIONS : ':' E X I S T E N T I A L '-' P R E C O N D I T I O N S ;
R_UNIVERSAL_PRECONDITIONS : ':' U N I V E R S A L '-' P R E C O N D I T I O N S ;
R_QUANTIFIED_PRECONDITIONS : ':' Q U A N T I F I E D '-' P R E C O N D I T I O N S ;
R_CONDITIONAL_EFFECTS : ':' C O N D I T I O N A L '-' E F F E C T S ;
R_FLUENTS : ':' F L U E N T S ;
R_ADL : ':' A D L ;
R_DURATIVE_ACTIONS : ':' D U R A T I V E '-' A C T I O N S ;
R_DERIVED_PREDICATES : ':' D E R I V E D '-' P R E D I C A T E S ;
R_TIMED_INITIAL_LITERALS : ':' T I M E D '-' I N I T I A L '-' L I T E R A L S ;
R_PREFERENCES : ':' P R E F E R E N C E S ;
R_CONSTRAINTS : ':' C O N S T R A I N T S ;
R_ACTION_COSTS : ':' A C T I O N '-' C O S T S ;
STR_NUMBER : N U M B E R ;

NAME
   : LETTER ANY_CHAR*
   ;

VARIABLE
   : '?' NAME
   ;

EQUALS
   : '='
   ;

NUMBER
   : DIGIT+ ( '.' DIGIT+ )?
   ;

LINE_COMMENT
   : ';' ~ ( '\n' | '\r' )* '\r'? '\n' -> skip
   ;

WHITESPACE
   : ( ' ' | '\t' | '\r' | '\n' )+ -> skip
   ;

fragment A:('a'|'A');
fragment B:('b'|'B');
fragment C:('c'|'C');
fragment D:('d'|'D');
fragment E:('e'|'E');
fragment F:('f'|'F');
fragment G:('g'|'G');
fragment H:('h'|'H');
fragment I:('i'|'I');
fragment J:('j'|'J');
fragment K:('k'|'K');
fragment L:('l'|'L');
fragment M:('m'|'M');
fragment N:('n'|'N');
fragment O:('o'|'O');
fragment P:('p'|'P');
fragment Q:('q'|'Q');
fragment R:('r'|'R');
fragment S:('s'|'S');
fragment T:('t'|'T');
fragment U:('u'|'U');
fragment V:('v'|'V');
fragment W:('w'|'W');
fragment X:('x'|'X');
fragment Y:('y'|'Y');
fragment Z:('z'|'Z');

fragment LETTER
   : A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z
   ;

fragment ANY_CHAR
   : LETTER | DIGIT | '-' | '_'
   ;
   
fragment DIGIT
   : '0' .. '9'
   ;
